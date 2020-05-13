# In this script we will be training and testing the AutoEncoder with multiple
# loss functions and parameter setups.

# Get access to parent folders
import os
import sys
sys.path.insert(0, '..')
import pickle

import torch
import torchvision

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import time
import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr

from Polytion import Generator as g
from Polytion import AutoEncoderNet as AE
from Polytion import prepData as prep
from Polytion import LossFunctions as lf

cuda = torch.cuda.is_available()
if cuda:
    print("cuda session enabled ..")
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("gpu session enabled ..")
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')

# Parameters:
batch_size = 32
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
TV_weight = 5.e-5
styleloss = lf.StyleLoss(1.0e-14)

num_epochs = 100

def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    optimizer_name = torch.optim.Adam
    lr = 0.0005
    w_decay = 0#1.0e-5
    optimizer = optimizer_name(model.parameters(), lr=lr, weight_decay=w_decay)
    gamma = 0.99
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma, last_epoch=-1)
    # Make info for suptitile
    info = str(layer)[27:-2] + ": N = " + str(N) +", r = " + str(rank) + ". " + str(optimizer_name)[25:-2] + " with " + str(scheduler)[26:-26]
    info += ", lr_init = " + str(lr) + ", w_decay = " + str(w_decay) +", gamma = " + str(gamma)
    
    start = time.time()
    epochs = tqdm.trange(num_epochs, desc="Start training", leave=True)
    try:
        for epoch in epochs:
            batch_loss = []
            for HiResIm, LoResIm in zip(HR_loader, LR_loader):   
                HiResIm = HiResIm.unsqueeze_(1).float()
                b, c, h, w = HiResIm.size()
                LoResIm = LoResIm.unsqueeze_(1).float()
                HiResIm = torch.autograd.Variable(HiResIm).to(device)
                LoResIm = torch.autograd.Variable(LoResIm).to(device)

                output = model(LoResIm).float()
                loss = lossfunc(output, HiResIm).float() #+ lf.TVLoss(TV_weight)(output).float() #+ styleloss(output.squeeze(1), HiResIm.squeeze(1)).float() #+ lf.TVLoss(TV_weight)(output).float()
                # loss /= (b*c*w*h)
                # loss /= w*h

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lossvalue = loss.item()
                all_loss.append(lossvalue)
                batch_loss.append(lossvalue)
                if torch.isnan(loss).sum() > 0:
                    print("nans in the loss function")
                    raise ValueError

            epoch_loss.append(np.mean(batch_loss))
            epochs.set_description("lr = {:.1e}, loss = {:.5e}".format(scheduler.get_lr()[0], epoch_loss[-1]))
            epochs.refresh()
            scheduler.step()

        print("Training finished, took ", round(time.time() - start,2), "s to complete ..")
    except (KeyboardInterrupt, SystemExit):
        print("\nscript execution halted ..")
        #print("loss = ", all_loss)
        sys.exit()
    except ValueError:
        print("\nnan found ..")
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
layer = g.PolyganCPlayer_seq
layerOptions = {'randnormweights':True, 'normalize':False}

fig, ax = plt.subplots(1,5, figsize=(27,5))
fs = 20
#fig.suptitle(info, fontsize=10)
ax[0].set_title("SL1 loss, $r$ = {}".format(rank), fontsize=fs)
ax[0].set_yscale('log')
ax[0].set_ylim([2.0e-3, 2])

ax[0].grid(True)
ax[0].set_xlabel('Epochs', fontsize=fs)
ax[0].set_ylabel('Training loss', fontsize=fs)

trainindex = 53
average = torch.zeros(HR_dim, HR_dim)
for HiResIm, LoResIm in zip(HR_loader, LR_loader):  
    average += HiResIm.mean(0)
    break
ax[1].imshow(average.cpu().detach().numpy(), cmap='gray')
ax[1].set_title("Average of training set", fontsize=fs)
ax[1].axis('off')

colors = ['b', 'r', 'g']
markers = ['x', 'o', '+']
lines = ['--', '-.', ':']
rank = 25
for i, N in enumerate([0, 1, 2]):
    model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
    epoch_loss, info = train(model)

    ax[0].plot(list(range(1,num_epochs+1)), epoch_loss, c=colors[i], ls=lines[i], label='$N$ = {}'.format(N))
    ax[0].scatter(list(range(1,num_epochs+1,4)), epoch_loss[::4], c=colors[i], marker=markers[i])
    ax[0].scatter(num_epochs, epoch_loss[-1], c=colors[i], marker=markers[i])
    model.eval()
    trainimage = testThisImage(train_LRimages[trainindex], model)
    psnrscore = psnr(train_HRimages[trainindex].astype(np.float), trainimage.astype(np.float))
    trainerror = epoch_loss[-1]

    ax[i+2].imshow(trainimage, cmap='gray')
    title = '$N$ = {}'.format(N)
    if i == 0:
        title = 'Bias'
    
    ax[i+2].set_title(title, fontsize=fs)
    ax[i+2].axes.xaxis.set_ticks([])#set_xticklabels([])
    ax[i+2].axes.yaxis.set_ticks([])#set_yticklabels([])
    ax[i+2].set_xlabel('$e$ = {:.2e}   PSNR = {:.2f}'.format(trainerror, psnrscore), fontsize=18)

ax[0].legend(prop={'size': 14})
fig.subplots_adjust(wspace=0.01, hspace=0.01)

timestamp = time.strftime("%d-%m-%Y_%H:%M:%S")
filename = "AutoEncoder/Nplot" + timestamp + ".png"
fig.savefig(filename, bbox_inches='tight')

