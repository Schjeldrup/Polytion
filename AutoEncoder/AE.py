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

from zipfile import ZipFile

from Polytion import Generator as g
from Polytion import AutoEncoderNet as AE
from Polytion import prepData as prep
from Polytion import LossFunctions as lf

cuda = torch.cuda.is_available()
if cuda:
    print("cuda session enabled")
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("gpu session enabled")
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')

# Parameters:
batch_size = 15
N = 5
rank = 20

LR_dim = 128
HR_dim = 512
bottleneck_dim = 32

scalefactor = HR_dim/bottleneck_dim
downscalefactor = bottleneck_dim/LR_dim

imagefolderpath = '000001_01_01'
HRpath = imagefolderpath + '/HRimages.pickle'
LRpath = imagefolderpath + '/LRimages.pickle'
if os.path.exists(HRpath):
    with open(HRpath, 'rb') as handle:
        HRimages = pickle.load(handle)
else:
    images = prep.load_images_from_folder(imagefolderpath)
    HRimages = prep.normalize_0(images)
    with open(HRpath, 'wb') as handle:
        pickle.dump(HRimages, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
if os.path.exists(LRpath):
    with open(LRpath, 'rb') as handle:
        LRimages = pickle.load(handle)
else:
    LRimages = prep.compress_images(HRimages)
    with open(LRpath, 'wb') as handle:
        pickle.dump(LRimages, handle)#, protocol=pickle.HIGHEST_PROTOCOL)

# images=prep.load_images_from_folder('/work3/projects/s181603-Jun-2020/Images_png.old/000020_03_01/')
# HRimages = prep.normalize_0(images)
# LRimages = prep.compress_images(HRimages)

HR_loader = torch.utils.data.DataLoader(HRimages,batch_size=batch_size)#, pin_memory=cuda)
LR_loader = torch.utils.data.DataLoader(LRimages, batch_size=batch_size)#, pin_memory=cuda)

#HR_loader = torch.utils.data.DataLoader(HRimages[:20],batch_size=batch_size) #pin_memory=cuda)
#LR_loader = torch.utils.data.DataLoader(LRimages[:20], batch_size=batch_size) #pin_memory=cuda)

# For quick testing:
# generatorOptions = {'parallel':False, 'workers':10}
# layerOptions = {'randnormweights':False, 'normalize':False, 'parallel':True}

# layer = g.PolyclassFTTlayer_seq
# #layer = g.PolyganCPlayer_seq
# model = Autoencoder_seq(layer, layerOptions, generatorOptions)

# for LoResIm in LR_loader:
#     LoResIm = LoResIm.unsqueeze_(1).float()
#     LoResIm = torch.autograd.Variable(LoResIm).to(device)
#     print("LoResIm = ", LoResIm.shape)
#     output = model(LoResIm).float()
#     break

# print("Succes")
# sys.exit()
#####

#lossfunc = torch.nn.SmoothL1Loss()
#lossfunc = torch.nn.L1Loss()
lossfunc = torch.nn.MSELoss()
TV_weight = 0

def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    optimizer_name = torch.optim.Adam
    lr = 0.01
    w_decay = 0
    optimizer = optimizer_name(model.parameters(), lr=lr, weight_decay=w_decay)
    gamma = 0.9
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    # Make info for suptitile
    info = str(layer)[27:-2] + ": " + str(optimizer_name)[25:-2] + " with " + str(scheduler)[26:-26]
    info += ", lr_init = " + str(lr) + ", w_decay = " + str(w_decay) +", gamma = " + str(gamma)

    num_epochs = 100
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
                loss = lossfunc(output, HiResIm).float() + lf.TVLoss(TV_weight)(output).float()
                loss /= (b*c*w*h)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossvalue = loss.item()
                all_loss.append(lossvalue)
                batch_loss.append(lossvalue)
                if torch.isnan(loss).sum() > 0:
                    raise ValueError

                epochs.set_description("lr = {:.1e}, loss = {:.5e}".format(scheduler.get_lr()[0], lossvalue))
                epochs.refresh()
                scheduler.step()

            epoch_loss.append(np.mean(batch_loss))
        print("Training finished, took ", round(time.time() - start,2), "s to complete")
    except (KeyboardInterrupt, SystemExit):
        print("\nscript execution halted ..")
        #print("loss = ", all_loss)
        sys.exit()
    except ValueError:
        print("\nnan found ..")
        #print("loss = ", all_loss)
        sys.exit()
    return epoch_loss, info


# ## 3. Training the different layers and generators:
#torch.autograd.set_detect_anomaly(True)
# For the old non sequential version:
# generatorOptions = {'parallel':False, 'workers':2}
# layerOptions = {'randnormweights':True, 'normalize':False, 'parallel':False}

generatorOptions = {}

layer = g.PolyclassFTTlayer_seq
layerOptions = {'randnormweights':True, 'normalize':False}

# layer = g.PolyganCPlayer_seq
# layerOptions = {'randnormweights':True, 'normalize':True}

model = AE.Autoencoder_seq(layer, N, rank, bottleneck_dim, HR_dim, downscalefactor, scalefactor, layerOptions, generatorOptions)

epoch_loss, info = train(model)

if epoch_loss[-1] == np.nan or epoch_loss[-1] == np.inf:
    print("Error: Inf or Nan found")
    sys.exit()

# Save and load the model as a test:
timestamp = time.strftime("%d-%m-%Y_%H:%M:%S")
modelname = "Testing/Trained_CP_model"
if os.path.exists(modelname):
    modelname += len[os.listdir()]
modelparamsname = modelname + "_params.pth"
modelname += ".pth"

modelparams = {"layer":layer, "N":N, "rank":rank, "bottleneck_dim":bottleneck_dim, "LR_dim":LR_dim, "HR_dim":HR_dim, "downscalefactor":downscalefactor, "scalefactor":scalefactor, "layerOptions":layerOptions, "generatorOptions":generatorOptions}
torch.save(modelparams, modelparamsname)
torch.save(model.state_dict(), modelname)
print("Saved model and params")

mp = torch.load(modelparamsname)
model = AE.Autoencoder_seq(mp["layer"], mp["N"], mp["rank"], mp["bottleneck_dim"], mp["HR_dim"], mp["downscalefactor"], mp["scalefactor"], mp["layerOptions"], mp["generatorOptions"])
model.load_state_dict(torch.load(modelname))
print("Loaded model and params")

fig, ax = plt.subplots(1,4, figsize=(20,5))
fig.suptitle(info, fontsize=10)
ax[0].plot(epoch_loss)
ax[0].set_yscale('log')
ax[0].grid(True)
lossfuncname = str(lossfunc)[0:-2] + " + " + str(TV_weight) + " * TV"
ax[0].set_title(lossfuncname + "loss")
ax[0].set_xlabel('epochs')

index = 1
ax[1].imshow(LRimages[index], cmap='gray')
ax[1].set_title("Input")
ax[1].axis('off')

model.eval()
test = torch.tensor(LRimages[index]).reshape(1,1,LR_dim,LR_dim)
if cuda:
    output = model(test).reshape(HR_dim,HR_dim).cpu().detach().numpy()
else:
    output = model(test).reshape(HR_dim,HR_dim).detach().numpy()

ax[2].imshow(output, cmap='gray')
ax[2].set_title("Output")
ax[2].axis('off')

ax[3].imshow(HRimages[index], cmap='gray')
ax[3].set_title("Truth")
ax[3].axis('off')

filename = "AutoEncoder/" + str(lossfunc)[0:-2] + timestamp + ".png"
fig.savefig(filename, bbox_inches='tight')
plt.show()

