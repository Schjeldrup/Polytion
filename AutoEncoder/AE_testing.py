# In this script we will be training and testing the AutoEncoder with multiple
# loss functions and parameter setups.

# Get access to parent folders
import os
import sys
#sys.path.insert(0, '..')
import pickle

import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm

import Generator as g
import prepData as prep
import ImageQualityAssesment as iqa

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
N = 6
rank = 10

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
        pickle.dump(HRimages, handle, protocol=pickle.HIGHEST_PROTOCOL)
if os.path.exists(LRpath):
    with open(LRpath, 'rb') as handle:
        LRimages = pickle.load(handle)
else:
    LRimages = prep.compress_images(HRimages)
    with open(LRpath, 'wb') as handle:
        pickle.dump(LRimages, handle, protocol=pickle.HIGHEST_PROTOCOL)


#HR_loader = torch.utils.data.DataLoader(HRimages,batch_size=batch_size, pin_memory=cuda)
#LR_loader = torch.utils.data.DataLoader(LRimages, batch_size=batch_size, pin_memory=cuda)
HR_loader = torch.utils.data.DataLoader(HRimages[:20],batch_size=batch_size, pin_memory=cuda)
LR_loader = torch.utils.data.DataLoader(LRimages[:20], batch_size=batch_size, pin_memory=cuda)

class Autoencoder(torch.nn.Module):
    def __init__(self, layer, layerOptions, generatorOptions):
        super(Autoencoder,self).__init__()
        self.encoder = g.Generator(layer, N, rank, bottleneck_dim, bottleneck_dim, downscalefactor, layerOptions, generatorOptions)
        self.decoder = g.Generator(layer, N, rank, HR_dim, HR_dim, scalefactor, layerOptions, generatorOptions)

    def forward(self, x):
        x = self.encoder(x.float())
        x = self.decoder(x)
        return x

MSE_lossfunc = torch.nn.MSELoss()
TV_lossfunc = iqa.TVLoss()
SSIM_lossfunc = iqa.SSIMLoss()
PSNR_lossfunc = iqa.PSNRLoss()
lossfunc = torch.nn.SmoothL1Loss()
lossfunc = iqa.TVLoss()

num_epochs = 15
def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = tqdm.trange(num_epochs, desc='Bar desc', leave=True)
    epochs.set_description("Start training")
    epochs.refresh()
    try:
        for epoch in epochs:
            batch_loss = []
            for HiResIm, LoResIm in zip(HR_loader, LR_loader):
                HiResIm = HiResIm.unsqueeze_(1).float()
                b, c, h, w = HiResIm.size()
                LoResIm = LoResIm.unsqueeze_(1).float()
                HiResIm = torch.autograd.Variable(HiResIm).to(device)
                LoResIm = LoResIm.to(device)

                output = model(LoResIm).float()
                extra_loss = torch.nn.SmoothL1Loss()(output, HiResIm).float()
                loss = lossfunc(output).float() + extra_loss / (b * c * h * w)
                #loss = -SSIM_lossfunc(output, HiResIm).float()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossvalue = loss.item()
                all_loss.append(lossvalue)
                batch_loss.append(lossvalue)
                if torch.isnan(loss).sum() > 0:
                    raise ValueError

                #epochs.set_description("Loss = {:.2e}".format(lossvalue))
                epochs.set_description("Loss = " + str(lossvalue))
                epochs.refresh()

            epoch_loss.append(np.mean(batch_loss))
        print("training finished")
    except (KeyboardInterrupt, SystemExit):
        print("\nscript execution halted ..")
    except ValueError:
        print("\nnan found ..")
    print("loss = ", all_loss)
    return epoch_loss


# ## 3. Training the different layers and generators:
generatorOptions = {'parallel':False, 'workers':0}
layerOptions = {'randnormweights':True, 'normalize':False, 'parallel':True}

#PolyganCPlayer:
# model = Autoencoder(g.PolyganCPlayer, layerOptions, generatorOptions)
#
# epoch_loss = train(model)
# plt.plot(epoch_loss)
# plt.savefig('PolyganCPlayer.png', bbox_inches='tight')
#
# model.eval()
# test = torch.tensor(LRimages[0]).reshape(1,1,LR_dim,LR_dim)
# output = model(test).reshape(HR_dim,HR_dim)
# torchvision.utils.save_image(output, "outputPolyganCPlayer.jpg")

# PolyclassFTTlayer:
model = Autoencoder(g.PolyganCPlayer, layerOptions, generatorOptions)
epoch_loss = train(model)
#epoch_loss = [0, 1, 2, 3]
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(epoch_loss)
ax[0].grid(True)
lossfuncname = str(lossfunc)[0:-2]
ax[0].set_title(lossfuncname + "loss")

model.eval()
test = torch.tensor(LRimages[0]).reshape(1,1,LR_dim,LR_dim)
output = model(test).reshape(HR_dim,HR_dim).detach().numpy()
ax[1].imshow(output, cmap='gray')
ax[1].axis('off')
#torchvision.utils.save_image(output, "outputPolyclassFTTlayer.jpg")

#filename = str(lossfunc)[0:-2] + datetime.now().strftime("%d/%m/%Y_%H_%M_%S") + '.png'
filename = "AutoEncoder/" + str(lossfunc)[0:-2] + time.strftime("%d-%m-%Y_%H:%M:%S") + ".png"
plt.savefig(filename, bbox_inches='tight')
