# In this script we will be training and testing the AutoEncoder with multiple
# loss functions and parameter setups.

# Get access to parent folders
import os
import sys
sys.path.insert(0, '..')
import pickle

import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from math import factorial
from Polytion import prepData as prep
from Polytion import LossFunctions as lf

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

cuda = False
if cuda:
    print("cuda session enabled")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("gpu session enabled")
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')


class PolyLayer_seq(torch.nn.Module):
    def __init__(self, N, rank, s, initmethod, biasgain, weightgain):
        # Start inherited structures:
        torch.nn.Module.__init__(self)
        # N = order of the polynomial = order of the tensor A:
        # A is of dimension (s, s, ..., s) = (N x s)
        # rank = rank used for the tensor cores
        # size = length of the flattened input
        # randnormweights = whether to use random weight initialization
        # normalize = whether to divide each weight by size self.s, as suggested by Morten
        self.N = N
        self.rank = rank
        self.s = s
        self.weightgain_bias = biasgain
        self.weightgain_weights = weightgain
        self.Laplace = torch.distributions.laplace.Laplace(0, 0.00005)
        self.eps = 0.07
        self.initweights = initmethod


class PolyganCPlayer_seq(PolyLayer_seq):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer_seq.__init__(self, N, rank, imwidth*imheight, layeroptions["initmethod"], layeroptions["biasgain"], layeroptions["weightgain"])

        # Initialize the bias
        b = torch.empty(self.s,1)
        if layeroptions["initmethod"] == "average":
            averagepath = '000001_01_01/average'
            with open(averagepath, 'rb') as handle:
                b = pickle.load(handle)
            self.initweights = torch.nn.init.xavier_normal_
        else: 
            print(self.initweights, self.weightgain_bias)
            self.initweights(b, self.weightgain_bias)
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Initialize the weights
        self.W = torch.nn.ParameterList()
        for n in range(N+1):
            factor_matrix = torch.zeros(self.s, self.rank)
            self.initweights(factor_matrix, self.weightgain_weights)
            self.W.append(torch.nn.Parameter(factor_matrix/(n+1)))

    def forward(self, z, b):
        Rsums = torch.zeros(b, 1, self.s)
        for n in range(self.N):
            f = torch.ones(b, 1, self.rank)
            for k in range(n+1):
                res = torch.matmul(z, self.W[k+1])
                f = f * res
            Rsums += torch.matmul(f, self.W[0].t()) / factorial(n + 1)
        return Rsums + self.b


class PolyclassFTTlayer_seq(PolyLayer_seq):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        PolyLayer_seq.__init__(self, N, rank, imwidth*imheight, layeroptions["initmethod"], layeroptions["biasgain"], layeroptions["weightgain"])

        self.ranklist = [1, 1]
        for n in range(self.N - 1):
            self.ranklist.insert(-1, self.rank)
        # Initialize the bias
        b = torch.empty(self.s,1)
        if layeroptions["initmethod"] == "average":
            averagepath = '000001_01_01/average'
            with open(averagepath, 'rb') as handle:
                b = pickle.load(handle)
            self.initweights = torch.nn.init.xavier_normal_
        else: 
            self.initweights(b, self.weightgain_bias)
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Start by making the tensor train: store the matrices in one big parameterlist

        self.TT = torch.nn.ParameterList()
        for n in range(self.N):
            # Make tensors of size (r_{k-1}, n_{k} = self.s, r_{k})
            TTcore = torch.zeros(self.ranklist[n], self.s, self.ranklist[n+1])
            self.initweights(TTcore, self.weightgain_weights)    
            self.TT.append(torch.nn.Parameter(TTcore))

    def forward(self, z, b):
        out = torch.zeros(b, self.s)
        for n in range(self.N):
            #print("n =",n)
            V = torch.zeros(b, 1, self.N, self.rank, self.rank)
            f = self.TT[0][0]
            #print("f1", f.shape)
            #for k in range(n+1):
            for k in range(self.N - n - 1, self.N):
                #print("k =",k)
                d1, d2 = self.ranklist[k], self.ranklist[k+1]
                #print(d1, d2)
                #print("z", z.shape)
                #print("TT", self.TT[k].permute(1,0,2).shape)
                V[:,:,k,0:d1,0:d2] = torch.matmul(z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2)).reshape(b, 1, d1, d2)
                #print(V[:,:,k,0:d1,0:d2].shape)
                f = torch.matmul(f, V[:, :, k,0:d1,0:d2])
            #print("f", f.reshape(b,1,-1).shape)
            #print("out", out.shape)
            out += f.reshape(b,-1) / factorial(n+1)
            #print("out", out.shape)
        return out + self.b
        

    def forwardOld(self, z, b):
        V = torch.zeros(b, 1, self.N, self.rank, self.rank)
        for k in range(self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            # print(self.TT[k].shape)
            # print(self.TT[k].permute(1,0,2).reshape(self.s, d1*d2).shape)
            V[:,:,k,0:d1,0:d2] = torch.matmul(z, self.TT[k].permute(1,0,2).reshape(self.s, d1*d2)).reshape(b, 1, d1, d2)

        f = self.TT[0][0]
        for k in range(self.N):
            d1, d2 = self.ranklist[k], self.ranklist[k+1]
            f = torch.matmul(f, V[:,:, k,0:d1,0:d2])
        return f.reshape(-1)
    

class Generator_seq(torch.nn.Module):
    def __init__(self, layer, N, rank, imwidth, imheight, scalefactor, layeroptions, generatoroptions):
        super(Generator_seq, self).__init__()
        # Channels: here we are working with greyscale images
        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        #self.PolyLayer = layer(N, rank, imwidth, imheight, layeroptions)
        #self.BN = torch.nn.BatchNorm1d(num_features=self.s)
        self.upsample = torch.nn.Upsample(scale_factor=scalefactor, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Register dimensions:
        xshape = x.shape
        if len(xshape) == 2:
            self.batch_size = 1
        else:
            self.batch_size = xshape[0]

        # Register x as attribute for parallel access, and clone because dataset would be overwritten
        self.x = self.upsample(x.float())
        # self.x = self.x.reshape(self.batch_size, self.s)
        # self.x = self.BN(self.x).unsqueeze(1)
        # self.x = self.PolyLayer(self.x, self.batch_size)
        # self.x = self.x.reshape(self.batch_size, self.c, self.imwidth, self.imheight)
        return self.x


# Parameters:
batch_size = 8
N = 1
rank = 50

LR_dim = 128
HR_dim = 512
bottleneck_dim = 32

scalefactor = HR_dim/bottleneck_dim
downscalefactor = bottleneck_dim/LR_dim

# Load the training set:
print("loading train and test sets ..")
Ntrain = 100
Ntest = 10
imagefolderpath = '000001_01_01'
HRpath = imagefolderpath + '/HRimages.pickle'
LRpath = imagefolderpath + '/LRimages.pickle'
if os.path.exists(HRpath):
    with open(HRpath, 'rb') as handle:
        HRimages = pickle.load(handle)
else:
    images = prep.load_images_from_all_folders('/work3/projects/s181603-Jun-2020/Images_png', Ntrain + Ntest)
    HRimages = prep.normalize_0(images)
    with open(HRpath, 'wb') as handle:
        pickle.dump(HRimages, handle)
if os.path.exists(LRpath):
    with open(LRpath, 'rb') as handle:
        LRimages = pickle.load(handle)
else:
    LRimages = prep.compress_images(HRimages)
    with open(LRpath, 'wb') as handle:
        pickle.dump(LRimages, handle)

train_HRimages = HRimages[0:Ntrain]
test_HRimages = HRimages[Ntrain:Ntrain+Ntest]
train_LRimages = LRimages[0:Ntrain]
test_LRimages = LRimages[Ntrain:Ntrain+Ntest]
print('{} training images'.format(len(train_HRimages)))
print('{} testing images'.format(len(test_HRimages)))

HR_loader = torch.utils.data.DataLoader(train_HRimages, shuffle=False, batch_size=batch_size)#, pin_memory=cuda)
LR_loader = torch.utils.data.DataLoader(train_LRimages, shuffle=False, batch_size=batch_size)#, pin_memory=cuda)

lossfunc = torch.nn.SmoothL1Loss()
#lossfunc = torch.nn.MSELoss()
TV_weight = 0#1.e-4
SL_weight = 0#1.e-10

num_epochs = 1
tvloss = lf.TVLoss(TV_weight)
styleloss = lf.StyleLoss(SL_weight)

def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    optimizer_name = torch.optim.Adam
    lr = 0.001
    w_decay = 0#1.0e-4
    optimizer = optimizer_name(model.parameters(), lr=lr, weight_decay=w_decay)
    gamma = 0.99
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=1.e-2)
    # Make info for suptitile
    info = str(layer)[27:-2] + ": N = " + str(N) +", r = " + str(rank) + ". " + str(optimizer_name)[25:-2] + " with " + str(scheduler)[26:-26]
    info += ", lr_init = " + str(lr) + ", w_decay = " + str(w_decay) +", gamma = " + str(gamma)
    info += ", TV = " + str(TV_weight) + ", Style = " + str(SL_weight)

    start = time.time()
    epochs = tqdm.trange(num_epochs, desc="Start training", leave=True)
    try:
        for epoch in epochs:
            batch_loss = []
            for HiResIm, LoResIm in zip(HR_loader, LR_loader):   
                HiResIm = HiResIm.unsqueeze_(1).float()
                b, c, h, w = HiResIm.size()
                HiResIm = torch.autograd.Variable(HiResIm).to(device)
                LoResIm = torch.autograd.Variable(LoResIm.unsqueeze(1)).to(device)

                output = model(LoResIm).float()
                loss = lossfunc(output, HiResIm).float() #+ styleloss(output.squeeze(1), HiResIm.squeeze(1)).float() #+ tvloss(output).float()

                
                a = list(model.parameters())[2].clone()

                lossvalue = loss.item() 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                b = list(model.parameters())[2].clone()
                
                result = torch.equal(a.data, b.data)


                all_loss.append(lossvalue)
                batch_loss.append(lossvalue)
                if torch.isnan(loss).sum() > 0:
                    print("nans in the loss function")
                    raise ValueError

            epoch_loss.append(np.mean(batch_loss))
            read_lr = scheduler.get_lr()[0]
            #read_lr = [group['lr'] for group in optimizer.param_groups][0]
            epochs.set_description("res = {}, lr = {:.1e}, loss = {:.5e}".format(result, read_lr, epoch_loss[-1]))
            epochs.refresh()
            scheduler.step()

        print("Training finished, took ", round(time.time() - start,2), "s to complete ..")
    except (KeyboardInterrupt, SystemExit):
        print("\nscript execution halted ..")
        #print("loss = ", all_loss)
        sys.exit()
    return epoch_loss, info

def testThisImage(image, model):
    test = torch.tensor(image).reshape(1,1,LR_dim,LR_dim)
    if cuda:
        return model(test).reshape(HR_dim,HR_dim).cpu().detach().numpy()
    else:
        return model(test).reshape(HR_dim,HR_dim).detach().numpy()

def testTheseImages(images, model):
    output = []
    for image in images:
        test = torch.tensor(image).reshape(1,1,LR_dim,LR_dim)
        if cuda:
            output.append(model(test).reshape(HR_dim,HR_dim).cpu().detach().numpy())
        else:
            output.append(model(test).reshape(HR_dim,HR_dim).detach().numpy())
    if len(output) == 1:
        return output[0]
    return output

# ## 3. Training the different layers and generators:
generatorOptions = {}
layerOptions = {'randnormweights':True, 'normalize':False}
layer = PolyganCPlayer_seq

model = Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)

fig, ax = plt.subplots(1,2, figsize=(12,6))
fs = 20
timestamp = time.strftime("%d-%m-%Y_%H:%M:%S")

trainindex = 2
ax[0].imshow(train_HRimages[trainindex], cmap='gray')
ax[0].set_title("Input from training set", fontsize=fs)
ax[0].axis('off')

trainimage = testThisImage(train_LRimages[trainindex], model)
trainimage_error = np.mean((trainimage - train_HRimages[trainindex])**2)
psnrscore = psnr(train_HRimages[trainindex].astype(np.float), trainimage.astype(np.float))
trainimages = testTheseImages(train_LRimages, model)
trainset_error = sum([np.mean((i - j)**2) for i,j in zip(trainimages, train_HRimages)])/len(trainimages)
ax[1].imshow(trainimage, cmap='gray')
ax[1].set_title('Trainset error = {:.2e}'.format(trainset_error), fontsize=fs)
ax[1].axes.xaxis.set_ticks([])#set_xticklabels([])
ax[1].axes.yaxis.set_ticks([])#set_yticklabels([])
ax[1].set_xlabel('$e$ = {:.2e}   PSNR = {:.2f}'.format(trainimage_error, psnrscore), fontsize=fs)

filename = "Just Generator/minerrortest" + timestamp + ".png"
fig.savefig(filename, bbox_inches='tight')

trainimages = testTheseImages(train_LRimages, model)
print(train_HRimages[0].shape)
trainset_error = sum([np.mean(np.sqrt((i - j)**2)) for i,j in zip(trainimages, train_HRimages)])/len(trainimages)
print(trainset_error)
trainset_error = sum([lf.smooth_l1(i,j) for i,j in zip(trainimages, train_HRimages)])/len(trainimages)
print(trainset_error)

testimages = testTheseImages(test_LRimages, model)
testset_error = sum([np.mean(np.sqrt((i - j)**2)) for i,j in zip(testimages, test_HRimages)])/len(testimages)
print(testset_error )
testset_error = sum([lf.smooth_l1(i,j) for i,j in zip(testimages, test_HRimages)])/len(testimages)
print(testset_error )
