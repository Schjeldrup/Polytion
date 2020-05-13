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

from math import factorial

from skimage.metrics import peak_signal_noise_ratio as psnr

from Polytion import Generator as g
from Polytion import AutoEncoderNet as AE
from Polytion import prepData as prep
from Polytion import LossFunctions as lf

cuda = torch.cuda.is_available()
if cuda:
    print("cuda session enabled")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("gpu session enabled")
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')

# Parameters:
batch_size = 16
N = 3
rank = 25

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


class PolyganCPlayer_seq(g.PolyLayer_seq):
    def __init__(self, N, rank, imwidth, imheight, layeroptions):
        g.PolyLayer_seq.__init__(self, N, rank, imwidth*imheight, randnormweights=layeroptions['randnormweights'])

        # Initialize the bias
        b = torch.empty(self.s,1)
        self.initweights(b, self.weightgain_bias/5)
        #torch.nn.init.normal_(b, mean=1.0e-5, std=0.0001)
        if layeroptions['normalize']:
            b = b / sqrt(self.s)
        self.b = torch.nn.Parameter(b.reshape(self.s))
        # Initialize the weights
        self.W = torch.nn.ParameterList()
        for n in range(N+1):
            factor_matrix = torch.zeros(self.s, self.rank)
            self.initweights(factor_matrix, self.weightgain_weights/5)
            #torch.nn.init.normal_(factor_matrix, mean=-1.0e-5, std=0.0008)
            
            #factor_matrix = self.Laplace.sample((self.s,self.rank))#.reshape(self.s, self.rank)
            self.W.append(torch.nn.Parameter(factor_matrix + self.eps))

    def forward(self, z, b, nceil = 4):
        Rsums = torch.zeros(b, 1, self.s)
        for n in range(nceil):
            f = torch.ones(b, 1, self.rank)
            for k in range(n+1):
                res = torch.matmul(z, self.W[k])
                f = f * res
            Rsums += torch.matmul(f, self.W[0].t()) / factorial(n + 1)
        return Rsums + self.b


class Generator_seq(torch.nn.Module):
    def __init__(self, layer, N, rank, imwidth, imheight, scalefactor, layeroptions, generatoroptions):
        super(Generator_seq, self).__init__()
        # Channels: here we are working with greyscale images
        self.c = 1
        self.imwidth, self.imheight = imwidth, imheight
        self.s = imwidth*imheight
        self.PolyLayer = layer(N, rank, imwidth, imheight, layeroptions)
        self.BN = torch.nn.BatchNorm2d(num_features=1)
        self.upsample = torch.nn.Upsample(scale_factor=scalefactor, mode='bilinear', align_corners=False)

    def forward(self, x, nceil):
        # Register dimensions:
        xshape = x.shape
        if len(xshape) == 2:
            self.batch_size = 1
        else:
            self.batch_size = xshape[0]

        # Register x as attribute for parallel access, and clone because dataset would be overwritten
        self.x = self.BN(x.float())
        self.x = self.upsample(self.x)
        self.x = self.x.reshape(self.batch_size, self.c, self.s)
        self.x = self.PolyLayer(self.x, self.batch_size, nceil)

        self.x = self.x.reshape(self.batch_size, self.c, self.imwidth, self.imheight)
        return self.x

lossfunc = torch.nn.SmoothL1Loss()
#lossfunc = torch.nn.MSELoss()
TV_weight = 0#1.e-4
SL_weight = 0#1.e-10

num_epochs = 50

tvloss = lf.TVLoss(TV_weight)
styleloss = lf.StyleLoss(SL_weight)

def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    optimizer_name = torch.optim.Adam
    lr = 0.01
    w_decay = 0#1.0e-4
    optimizer = optimizer_name(model.parameters(), lr=lr, weight_decay=w_decay)
    gamma = 0.95
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, threshold=1.e-2)
    # Make info for suptitile
    info = str(layer)[27:-2] + ": N = " + str(N) +", r = " + str(rank) + ". " + str(optimizer_name)[25:-2] + " with " + str(scheduler)[26:-26]
    info += ", lr_init = " + str(lr) + ", w_decay = " + str(w_decay) +", gamma = " + str(gamma)
    info += ", TV = " + str(TV_weight) + ", Style = " + str(SL_weight)

    limits = [int(0.1*num_epochs), int(0.4*num_epochs), int(0.6*num_epochs), int(0.8*num_epochs), num_epochs]
    nceil = 0

    start = time.time()
    epochs = tqdm.trange(num_epochs, desc="Start training", leave=True)
    try:
        for epoch in epochs:
            batch_loss = []
            for HiResIm, LoResIm in zip(HR_loader, LR_loader): 

                # if epoch >= limits[nceil]:
                #     # for name, param in model.named_parameters():
                #     #     if 'BN' in name:
                #     #         continue
                    
                #     paramsbefore = [param for name, param in model.named_parameters() if 'BN' not in name]
                #     params = paramsbefore
                #     params[nceil + 1] = params[nceil]
                #     nceil += 1    
                nceil = 4       
          
                HiResIm = HiResIm.unsqueeze_(1).float()
                b, c, h, w = HiResIm.size()
                LoResIm = LoResIm.unsqueeze_(1).float()
                HiResIm = torch.autograd.Variable(HiResIm).to(device)
                LoResIm = torch.autograd.Variable(LoResIm).to(device)

                output = model(LoResIm, nceil).float()
                loss = lossfunc(output, HiResIm).float() #+ styleloss(output.squeeze(1), HiResIm.squeeze(1)).float() #+ tvloss(output).float()

                lossvalue = loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                all_loss.append(lossvalue)
                batch_loss.append(lossvalue)
                if torch.isnan(loss).sum() > 0:
                    print("nans in the loss function")
                    raise ValueError

            epoch_loss.append(np.mean(batch_loss))
            read_lr = scheduler.get_lr()[0]
            #read_lr = [group['lr'] for group in optimizer.param_groups][0]
            epochs.set_description("nceil = {}, lr = {:.1e}, loss = {:.5e}".format(nceil, read_lr, epoch_loss[-1]))
            epochs.refresh()
            scheduler.step()

        print("Training finished, took ", round(time.time() - start,2), "s to complete ..")
    except (KeyboardInterrupt, SystemExit):
        print("\nscript execution halted ..")
        #print("loss = ", all_loss)
        sys.exit()
    return epoch_loss, info

nceil = N + 1

def testThisImage(image, model):
    test = torch.tensor(image).reshape(1,1,LR_dim,LR_dim)
    if cuda:
        return model(test, nceil).reshape(HR_dim,HR_dim).cpu().detach().numpy()
    else:
        return model(test, nceil).reshape(HR_dim,HR_dim).detach().numpy()

def testTheseImages(images, model):
    output = []
    for image in images:
        test = torch.tensor(image).reshape(1,1,LR_dim,LR_dim)
        if cuda:
            output.append(model(test, nceil).reshape(HR_dim,HR_dim).cpu().detach().numpy())
        else:
            output.append(model(test, nceil).reshape(HR_dim,HR_dim).detach().numpy())
    if len(output) == 1:
        return output[0]
    return output

# ## 3. Training the different layers and generators:
generatorOptions = {}
layer = PolyganCPlayer_seq
layerOptions = {'randnormweights':True, 'normalize':False}

plt.figure(figsize=(8,6))
#model = AE.Autoencoder_seq(layer, N, rank, bottleneck_dim, HR_dim, downscalefactor, scalefactor, layerOptions, generatorOptions)
model = Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
for name, param in model.named_parameters():
    if 'BN' in name:
        continue
    vector = param.reshape(-1).cpu().detach().numpy()
    hist, edges = np.histogram(vector, density=True, bins = int(np.sqrt(vector.shape[0])/2))
    bin_centers = 0.5*(edges[1:] + edges[:-1])
    plt.plot(bin_centers, hist, label=name + 'init')


# print("Running on ", torch.cuda.device_count(), "nodes")
# model = torch.nn.DataParallel(model)

epoch_loss, info = train(model)

timestamp = time.strftime("%d-%m-%Y_%H:%M:%S")
# Now get weights for every layer:
for name, param in model.named_parameters():
    if 'BN' in name:
        continue
    vector = param.reshape(-1).cpu().detach().numpy()
    print(name, vector.mean(), vector.std())
    hist, edges = np.histogram(vector, density=True, bins = int(np.sqrt(vector.shape[0])/2))
    bin_centers = 0.5*(edges[1:] + edges[:-1])
    plt.plot(bin_centers, hist, label=name)
plt.xlim([-0.005, 0.005])
plt.grid(True)
plt.legend()
plt.xlabel('Weight value', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title("Network weights", fontsize=20)
filename = "Just Generator/GENWEIGHTS" + timestamp + ".png"
plt.savefig(filename, bbox_inches='tight')

fig, ax = plt.subplots(1,5, figsize=(30,6))
fs = 20
fig.suptitle(info, fontsize=10)
ax[0].plot(list(range(1,num_epochs+1)), epoch_loss, c='b', label='trainloss')
ax[0].set_title("{}, $N$ = {}, $r$ = {}".format(lossfunc, N, rank), fontsize=fs)
ax[0].set_yscale('log')
ax[0].grid(True)
ax[0].set_xlabel('Training epochs', fontsize=fs)
ax[0].set_ylabel('Loss', fontsize=fs)
ax[0].legend()

trainindex = 75
ax[1].imshow(train_HRimages[trainindex], cmap='gray')
ax[1].set_title("Input from training set", fontsize=fs)
ax[1].axis('off')

trainimage = testThisImage(train_LRimages[trainindex], model)
trainimage_error = np.mean((trainimage - train_HRimages[trainindex])**2)
psnrscore = psnr(train_HRimages[trainindex].astype(np.float), trainimage.astype(np.float))
trainset_error = epoch_loss[-1]
ax[2].imshow(trainimage, cmap='gray')
ax[2].set_title('Prediction, trainset error = {:.2e}'.format(trainset_error), fontsize=fs)
ax[2].axes.xaxis.set_ticks([])#set_xticklabels([])
ax[2].axes.yaxis.set_ticks([])#set_yticklabels([])
ax[2].set_xlabel('$e$ = {:.2e}   PSNR = {:.2f}'.format(trainimage_error, psnrscore), fontsize=fs)


testindex = 8
ax[3].imshow(test_HRimages[testindex], cmap='gray')
ax[3].set_title("Input from test set", fontsize=fs)
ax[3].axis('off')

testimage = testThisImage(test_LRimages[testindex], model)
testimage_error = np.mean((testimage - test_HRimages[testindex])**2)
psnrscore = psnr(test_HRimages[testindex].astype(np.float), testimage.astype(np.float))
testimages = testTheseImages(test_LRimages, model)
testset_error = sum([np.mean((i - j)**2) for i,j in zip(testimages, test_HRimages[testindex])])/2
ax[4].imshow(testimage, cmap='gray')
ax[4].set_title('Prediction, testset error = {:.2e}'.format(testset_error), fontsize=fs)
ax[4].axes.xaxis.set_ticks([])#set_xticklabels([])
ax[4].axes.yaxis.set_ticks([])#set_yticklabels([])
ax[4].set_xlabel('$e$ = {:.2e}   PSNR = {:.2f}'.format(testimage_error, psnrscore), fontsize=fs)

# average = torch.zeros(HR_dim, HR_dim)
# for HiResIm, LoResIm in zip(HR_loader, LR_loader):  
#     average += HiResIm.mean(0)


fig.subplots_adjust(wspace=0.01, hspace=0.01)

filename = "Just Generator/" + timestamp + ".png"
fig.savefig(filename, bbox_inches='tight')

