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

from Polytion import Generator as g
from Polytion.Polynomial_generator_exploration import Generator as gold
from Polytion.Polynomial_generator_exploration import SequentialGenerator as sgold
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

#images=prep.load_images_from_folder('/work3/projects/s181603-Jun-2020/Images_png.old/000020_02_01/')
#HRimages = prep.normalize_0(images)
#LRimages = prep.compress_images(HRimages)

HR_loader = torch.utils.data.DataLoader(HRimages,batch_size=batch_size)#, pin_memory=cuda)
LR_loader = torch.utils.data.DataLoader(LRimages, batch_size=batch_size)#, pin_memory=cuda)
#HR_loader = torch.utils.data.DataLoader(HRimages[:20],batch_size=batch_size) #pin_memory=cuda)
#LR_loader = torch.utils.data.DataLoader(LRimages[:20], batch_size=batch_size) #pin_memory=cuda)

class goldAutoencoder(torch.nn.Module):
    def __init__(self, layer, layerOptions, generatorOptions):
        super(goldAutoencoder,self).__init__()
        
        self.encoder = sgold.Generator(layer, N, rank, bottleneck_dim, bottleneck_dim, downscalefactor)
        self.decoder = sgold.Generator(layer, N, rank, HR_dim, HR_dim, scalefactor)

    def forward(self, x):
        x = self.encoder(x.float())
        x = self.decoder(x)
        return x

class Autoencoder(torch.nn.Module):
    def __init__(self, layer, layerOptions, generatorOptions):
        super(Autoencoder,self).__init__()
        self.encoder = g.Generator(layer, N, rank, bottleneck_dim, bottleneck_dim, downscalefactor, layerOptions, generatorOptions)
        self.decoder = g.Generator(layer, N, rank, HR_dim, HR_dim, scalefactor, layerOptions, generatorOptions)
        
    def forward(self, x):
        x = self.encoder(x.float())
        x = self.decoder(x)
        return x

lossfunc = torch.nn.SmoothL1Loss()
#lossfunc = torch.nn.L1Loss()
lossfunc = torch.nn.MSELoss()
lr = 0.002

num_epochs = 200
def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay= 5e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.2)

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
                loss = lossfunc(output, HiResIm).float()/(b*c*w*h) + lf.TVLoss()(output).float()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossvalue = loss.item()
                all_loss.append(lossvalue)
                batch_loss.append(lossvalue)
                if torch.isnan(loss).sum() > 0:
                    raise ValueError

                epochs.set_description("Loss = " + str(lossvalue))
                epochs.refresh()

            epoch_loss.append(np.mean(batch_loss))
        print("training finished")
    except (KeyboardInterrupt, SystemExit):
        print("\nscript execution halted ..")
        print("loss = ", all_loss)
        sys.exit()
    except ValueError:
        print("\nnan found ..")
        print("loss = ", all_loss)
        sys.exit()
    #print("loss = ", all_loss)
    return epoch_loss


# ## 3. Training the different layers and generators:
generatorOptions = {'parallel':False, 'workers':10}
layerOptions = {'randnormweights':True, 'normalize':True, 'parallel':False}

#layer = gold.PolyGAN_CP_Layer
#model = goldAutoencoder(layer, layerOptions, generatorOptions)

#layer = g.PolyclassFTTlayer # normalizations doesn't work, don't use TVloss
layer = g.PolyganCPlayer
model = Autoencoder(layer, layerOptions, generatorOptions)
epoch_loss = train(model)
#epoch_loss = [0, 1, 2, 3]

if epoch_loss[-1] == np.nan or epoch_loss[-1] == np.inf:
    sys.exit()

fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].plot(epoch_loss)
ax[0].grid(True)
lossfuncname = str(lossfunc)[0:-2]
ax[0].set_title(lossfuncname + "loss, lr = " + str(lr) + " " + str(layer))
ax[0].set_xlabel('epochs')

model.eval()
test = torch.tensor(LRimages[0]).reshape(1,1,LR_dim,LR_dim)
if cuda:
    output = model(test).reshape(HR_dim,HR_dim).cpu().detach().numpy()
else:
    output = model(test).reshape(HR_dim,HR_dim).detach().numpy()

ax[1].imshow(output, cmap='gray')
ax[1].set_title("Output")
ax[1].axis('off')

ax[2].imshow(HRimages[0], cmap='gray')
ax[2].grid(True)
ax[2].set_title("Truth")
ax[2].axis('off')
#torchvision.utils.save_image(output, "outputPolyclassFTTlayer.jpg")

#filename = str(lossfunc)[0:-2] + datetime.now().strftime("%d/%m/%Y_%H_%M_%S") + '.png'
filename = "AutoEncoder/" + str(lossfunc)[0:-2] + time.strftime("%d-%m-%Y_%H:%M:%S") + ".png"
plt.savefig(filename, bbox_inches='tight')


