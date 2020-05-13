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

from Polytion import Generator as g
from Polytion import AutoEncoderNet as AE
from Polytion import prepData as prep
from Polytion import LossFunctions as lf

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

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
SL_weight = 0#1.e-4

num_epochs = 50
tvloss = lf.TVLoss(TV_weight)
styleloss = lf.StyleLoss(SL_weight)

def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    epoch_psnr = []
    all_loss = []
    optimizer_name = torch.optim.Adam
    #lr = 0.001
    w_decay = 0#1.0e-4
    optimizer = optimizer_name(model.parameters(), lr=lr, weight_decay=w_decay)
    gamma = 0.97
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    # Make info for suptitile
    # info = str(layer)[27:-2] + ": N = " + str(N) +", r = " + str(rank) + ". " + str(optimizer_name)[25:-2] + " with " + str(scheduler)[26:-26]
    # info += ", lr_init = " + str(lr) + ", w_decay = " + str(w_decay) +", gamma = " + str(gamma)
    # info += ", TV = " + str(TV_weight) + ", Style = " + str(SL_weight)

    start = time.time()
    epochs = tqdm.trange(num_epochs, desc="Start training", leave=True)
    try:
        for epoch in epochs:
            batch_loss = []
            batch_psnr = []
            for HiResIm, LoResIm in zip(HR_loader, LR_loader):   
                HiResIm = HiResIm.unsqueeze_(1).float()
                b, c, h, w = HiResIm.size()
                LoResIm = LoResIm.unsqueeze_(1).float()
                HiResIm = torch.autograd.Variable(HiResIm).to(device)
                LoResIm = torch.autograd.Variable(LoResIm).to(device)

                output = model(LoResIm).float()
                loss = lossfunc(output, HiResIm).float() + styleloss(output.squeeze(1), HiResIm.squeeze(1)).float() + tvloss(output).float()
                
                current_psnr = psnr(HiResIm.cpu().detach().numpy(), output.cpu().detach().numpy())
                batch_psnr.append(current_psnr)

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
            epoch_psnr.append(sum(batch_psnr)/len(batch_psnr))
            read_lr = scheduler.get_lr()[0]
            #read_lr = [group['lr'] for group in optimizer.param_groups][0]
            epochs.set_description("lr = {:.1e}, loss = {:.5e}".format(read_lr, epoch_loss[-1]))
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
    return epoch_loss, epoch_psnr

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

timestamp = time.strftime("%d-%m-%Y_%H:%M:%S")
fig, ax = plt.subplots(2,4, figsize=(30,8))

ax[0,0].tick_params(labelbottom=False)
for i in range(1,4):
    ax[0,i].tick_params(labelbottom=False)
    ax[0,i].tick_params(labelleft=False)
    ax[1,i].tick_params(labelleft=False)


cmap = plt.cm.gnuplot

nlines = 8
#line_colors = cmap(np.linspace(0,1,nlines))
line_colors = ['k', 'y', 'b', 'r', 'g', 'm']
markers = ['x', 'o', '+', 'D', '.', '*']

fs = 20
num_epochs = 50
xvals = list(range(1,num_epochs+1))
xvalsf = list(range(1,num_epochs+1,5))


# Plot 1:
lossfunc = torch.nn.L1Loss()
lr = 0.001

# w_decay = 1.0e-4
# model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
# epoch_loss = train(model)
# ax[0,0].plot(xvals, epoch_loss, c=line_colors[1])
# ax[0,0].scatter(xvalsf, epoch_loss[::4], c=line_colors[1], marker=markers[1], label='W decay: {:.1e}'.format(w_decay))
# ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[1], marker=markers[1])
# w_decay = 0

w_decay = 1.0e-3
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,0].plot(xvals, epoch_loss, c=line_colors[2])
ax[0,0].scatter(xvalsf, epoch_loss[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[2], marker=markers[2])

ax[1,0].plot(xvals, epoch_psnr, c=line_colors[2])
ax[1,0].scatter(xvalsf, epoch_psnr[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[1,0].scatter(num_epochs, epoch_psnr[-1], c=line_colors[2], marker=markers[2])
w_decay = 0

TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,0].plot(xvals, epoch_loss, c=line_colors[3])
ax[0,0].scatter(xvalsf, epoch_loss[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[3], marker=markers[3])

ax[1,0].plot(xvals, epoch_psnr, c=line_colors[3])
ax[1,0].scatter(xvalsf, epoch_psnr[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[1,0].scatter(num_epochs, epoch_psnr[-1], c=line_colors[3], marker=markers[3])
tvloss = lf.TVLoss(0.0)

st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,0].plot(xvals, epoch_loss, c=line_colors[4])
ax[0,0].scatter(xvalsf, epoch_loss[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[4], marker=markers[4])

ax[1,0].plot(xvals, epoch_psnr, c=line_colors[4])
ax[1,0].scatter(xvalsf, epoch_psnr[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[1,0].scatter(num_epochs, epoch_psnr[-1], c=line_colors[4], marker=markers[4])
styleloss = lf.StyleLoss(0.0)

w_decay = 1.0e-3
TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,0].plot(xvals, epoch_loss, c=line_colors[5])
ax[0,0].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])

ax[1,0].plot(xvals, epoch_loss, c=line_colors[5])
ax[1,0].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[1,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])
w_decay = 0
TV_weight = 0
tvloss = lf.TVLoss(TV_weight)
st_weight = 0
styleloss = lf.StyleLoss(SL_weight)

model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,0].plot(xvals, epoch_loss, c=line_colors[0])
ax[0,0].scatter(xvalsf, epoch_loss[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[0], marker=markers[0],s=45)

ax[1,0].plot(xvals, epoch_psnr, c=line_colors[0])
ax[1,0].scatter(xvalsf, epoch_psnr[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[1,0].scatter(num_epochs, epoch_psnr[-1], c=line_colors[0], marker=markers[0],s=45)

ax[0,0].set_title("L1 Loss", fontsize=fs)
ax[0,0].set_yscale('log')
ax[0,0].grid(True)
ax[0,0].set_ylim([2.e-3, 1])
#ax[0,0].set_xlabel('Training epochs', fontsize=fs)
ax[0,0].set_ylabel('Training loss', fontsize=fs)

ax[1,0].grid(True)
ax[1,0].set_yscale('linear')
ax[1,0].set_ylim([0, 25])
ax[1,0].set_xlabel('Epochs', fontsize=fs)
ax[1,0].set_ylabel('PSNR', fontsize=fs)
ax[1,0].legend(prop={'size': 14})



# Plot 2:
lossfunc = lf.CharbonnierLoss()
# w_decay = 1.0e-4
# model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
# epoch_loss = train(model)
# ax[0,0].plot(xvals, epoch_loss, c=line_colors[1])
# ax[0,0].scatter(xvalsf, epoch_loss[::4], c=line_colors[1], marker=markers[1], label='W decay: {:.1e}'.format(w_decay))
# ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[1], marker=markers[1])
# w_decay = 0

w_decay = 1.0e-3
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,1].plot(xvals, epoch_loss, c=line_colors[2])
ax[0,1].scatter(xvalsf, epoch_loss[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[0,1].scatter(num_epochs, epoch_loss[-1], c=line_colors[2], marker=markers[2])

ax[1,1].plot(xvals, epoch_psnr, c=line_colors[2])
ax[1,1].scatter(xvalsf, epoch_psnr[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[1,1].scatter(num_epochs, epoch_psnr[-1], c=line_colors[2], marker=markers[2])
w_decay = 0

TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,1].plot(xvals, epoch_loss, c=line_colors[3])
ax[0,1].scatter(xvalsf, epoch_loss[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[0,1].scatter(num_epochs, epoch_loss[-1], c=line_colors[3], marker=markers[3])

ax[1,1].plot(xvals, epoch_psnr, c=line_colors[3])
ax[1,1].scatter(xvalsf, epoch_psnr[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[1,1].scatter(num_epochs, epoch_psnr[-1], c=line_colors[3], marker=markers[3])
tvloss = lf.TVLoss(0.0)

st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,1].plot(xvals, epoch_loss, c=line_colors[4])
ax[0,1].scatter(xvalsf, epoch_loss[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[0,1].scatter(num_epochs, epoch_loss[-1], c=line_colors[4], marker=markers[4])

ax[1,1].plot(xvals, epoch_psnr, c=line_colors[4])
ax[1,1].scatter(xvalsf, epoch_psnr[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[1,1].scatter(num_epochs, epoch_psnr[-1], c=line_colors[4], marker=markers[4])
styleloss = lf.StyleLoss(0.0)

w_decay = 1.0e-3
TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,0].plot(xvals, epoch_loss, c=line_colors[5])
ax[0,0].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])

ax[1,1].plot(xvals, epoch_loss, c=line_colors[5])
ax[1,1].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[1,1].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])
w_decay = 0
TV_weight = 0
tvloss = lf.TVLoss(TV_weight)
st_weight = 0
styleloss = lf.StyleLoss(SL_weight)

model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,1].plot(xvals, epoch_loss, c=line_colors[0])
ax[0,1].scatter(xvalsf, epoch_loss[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[0,1].scatter(num_epochs, epoch_loss[-1], c=line_colors[0], marker=markers[0],s=45)

ax[1,1].plot(xvals, epoch_psnr, c=line_colors[0])
ax[1,1].scatter(xvalsf, epoch_psnr[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[1,1].scatter(num_epochs, epoch_psnr[-1], c=line_colors[0], marker=markers[0],s=45)


ax[0,1].set_title("Charbonnier Loss", fontsize=fs)
ax[0,1].set_yscale('log')
ax[0,1].grid(True)
ax[0,1].set_ylim([2.e-3, 1])

ax[1,1].grid(True)
ax[1,1].set_yscale('linear')
ax[1,1].set_ylim([0, 25])
ax[1,1].set_xlabel('Epochs', fontsize=fs)




# Plot 3:
lossfunc = torch.nn.SmoothL1Loss()

# w_decay = 1.0e-4
# model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
# epoch_loss = train(model)
# ax[0,0].plot(xvals, epoch_loss, c=line_colors[1])
# ax[0,0].scatter(xvalsf, epoch_loss[::4], c=line_colors[1], marker=markers[1], label='W decay: {:.1e}'.format(w_decay))
# ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[1], marker=markers[1])
# w_decay = 0

w_decay = 1.0e-3
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,2].plot(xvals, epoch_loss, c=line_colors[2])
ax[0,2].scatter(xvalsf, epoch_loss[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[0,2].scatter(num_epochs, epoch_loss[-1], c=line_colors[2], marker=markers[2])

ax[1,2].plot(xvals, epoch_psnr, c=line_colors[2])
ax[1,2].scatter(xvalsf, epoch_psnr[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[1,2].scatter(num_epochs, epoch_psnr[-1], c=line_colors[2], marker=markers[2])
w_decay = 0

TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,2].plot(xvals, epoch_loss, c=line_colors[3])
ax[0,2].scatter(xvalsf, epoch_loss[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[0,2].scatter(num_epochs, epoch_loss[-1], c=line_colors[3], marker=markers[3])

ax[1,2].plot(xvals, epoch_psnr, c=line_colors[3])
ax[1,2].scatter(xvalsf, epoch_psnr[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[1,2].scatter(num_epochs, epoch_psnr[-1], c=line_colors[3], marker=markers[3])
tvloss = lf.TVLoss(0.0)

st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,2].plot(xvals, epoch_loss, c=line_colors[4])
ax[0,2].scatter(xvalsf, epoch_loss[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[0,2].scatter(num_epochs, epoch_loss[-1], c=line_colors[4], marker=markers[4])

ax[1,2].plot(xvals, epoch_psnr, c=line_colors[4])
ax[1,2].scatter(xvalsf, epoch_psnr[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[1,2].scatter(num_epochs, epoch_psnr[-1], c=line_colors[4], marker=markers[4])
styleloss = lf.StyleLoss(0.0)

w_decay = 1.0e-3
TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,2].plot(xvals, epoch_loss, c=line_colors[5])
ax[0,2].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[0,2].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])

ax[1,2].plot(xvals, epoch_loss, c=line_colors[5])
ax[1,2].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[1,2].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])
w_decay = 0
TV_weight = 0
tvloss = lf.TVLoss(TV_weight)
st_weight = 0
styleloss = lf.StyleLoss(SL_weight)

model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,2].plot(xvals, epoch_loss, c=line_colors[0])
ax[0,2].scatter(xvalsf, epoch_loss[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[0,2].scatter(num_epochs, epoch_loss[-1], c=line_colors[0], marker=markers[0],s=45)

ax[1,2].plot(xvals, epoch_psnr, c=line_colors[0])
ax[1,2].scatter(xvalsf, epoch_psnr[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[1,2].scatter(num_epochs, epoch_psnr[-1], c=line_colors[0], marker=markers[0],s=45)

ax[0,2].set_title("Smooth L1 Loss", fontsize=fs)
ax[0,2].set_yscale('log')
ax[0,2].grid(True)
ax[0,2].set_ylim([2.e-3, 1])

ax[1,2].grid(True)
ax[1,2].set_yscale('linear')
ax[1,2].set_ylim([0, 25])
ax[1,2].set_xlabel('Epochs', fontsize=fs)




# Plot 3:
lossfunc = torch.nn.MSELoss()

# w_decay = 1.0e-4
# model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
# epoch_loss = train(model)
# ax[0,0].plot(xvals, epoch_loss, c=line_colors[1])
# ax[0,0].scatter(xvalsf, epoch_loss[::4], c=line_colors[1], marker=markers[1], label='W decay: {:.1e}'.format(w_decay))
# ax[0,0].scatter(num_epochs, epoch_loss[-1], c=line_colors[1], marker=markers[1])
# w_decay = 0

w_decay = 1.0e-3
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,3].plot(xvals, epoch_loss, c=line_colors[2])
ax[0,3].scatter(xvalsf, epoch_loss[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[0,3].scatter(num_epochs, epoch_loss[-1], c=line_colors[2], marker=markers[2])

ax[1,3].plot(xvals, epoch_psnr, c=line_colors[2])
ax[1,3].scatter(xvalsf, epoch_psnr[::5], c=line_colors[2], marker=markers[2], label='W decay: {:.1e}'.format(w_decay))
ax[1,3].scatter(num_epochs, epoch_psnr[-1], c=line_colors[2], marker=markers[2])
w_decay = 0

TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,3].plot(xvals, epoch_loss, c=line_colors[3])
ax[0,3].scatter(xvalsf, epoch_loss[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[0,3].scatter(num_epochs, epoch_loss[-1], c=line_colors[3], marker=markers[3])

ax[1,3].plot(xvals, epoch_psnr, c=line_colors[3])
ax[1,3].scatter(xvalsf, epoch_psnr[::5], c=line_colors[3], marker=markers[3], label='TV loss')
ax[1,3].scatter(num_epochs, epoch_psnr[-1], c=line_colors[3], marker=markers[3])
tvloss = lf.TVLoss(0.0)

st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,3].plot(xvals, epoch_loss, c=line_colors[4])
ax[0,3].scatter(xvalsf, epoch_loss[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[0,3].scatter(num_epochs, epoch_loss[-1], c=line_colors[4], marker=markers[4])

ax[1,3].plot(xvals, epoch_psnr, c=line_colors[4])
ax[1,3].scatter(xvalsf, epoch_psnr[::5], c=line_colors[4], marker=markers[4], label='Texture loss')
ax[1,3].scatter(num_epochs, epoch_psnr[-1], c=line_colors[4], marker=markers[4])
styleloss = lf.StyleLoss(0.0)

w_decay = 1.0e-3
TV_weight = 2.0e-5
tvloss = lf.TVLoss(TV_weight)
st_weight = 1.0e-10
styleloss = lf.StyleLoss(SL_weight)
model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,3].plot(xvals, epoch_loss, c=line_colors[5])
ax[0,3].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[0,3].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])

ax[1,3].plot(xvals, epoch_loss, c=line_colors[5])
ax[1,3].scatter(xvalsf, epoch_loss[::5], c=line_colors[5], marker=markers[5], label='All')
ax[1,3].scatter(num_epochs, epoch_loss[-1], c=line_colors[5], marker=markers[5])
w_decay = 0
TV_weight = 0
tvloss = lf.TVLoss(TV_weight)
st_weight = 0
styleloss = lf.StyleLoss(SL_weight)

model = g.Generator_seq(layer, N, rank, HR_dim, HR_dim, 4, layerOptions, generatorOptions)
epoch_loss, epoch_psnr = train(model)
ax[0,3].plot(xvals, epoch_loss, c=line_colors[0])
ax[0,3].scatter(xvalsf, epoch_loss[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[0,3].scatter(num_epochs, epoch_loss[-1], c=line_colors[0], marker=markers[0], s=45)

ax[1,3].plot(xvals, epoch_psnr, c=line_colors[0])
ax[1,3].scatter(xvalsf, epoch_psnr[::5], c=line_colors[0], marker=markers[0], label='No regularization')
ax[1,3].scatter(num_epochs, epoch_psnr[-1], c=line_colors[0], marker=markers[0], s=45)

ax[0,3].set_title("MSE Loss", fontsize=fs)
ax[0,3].set_yscale('log')
ax[0,3].grid(True)
ax[0,3].set_ylim([2.e-3, 1])

ax[1,3].grid(True)
ax[1,3].set_yscale('linear')
ax[1,3].set_ylim([0, 25])
ax[1,3].set_xlabel('Epochs', fontsize=fs)



fig.subplots_adjust(wspace=0.02, hspace=0.04)
filename = "Just Generator/lossplot" + timestamp + ".png"
fig.savefig(filename, bbox_inches='tight')

