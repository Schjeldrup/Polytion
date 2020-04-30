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
batch_size = 100
N = 6
rank = 10

LR_dim = 14
HR_dim = 28
bottleneck_dim = 8

scalefactor = HR_dim/bottleneck_dim
downscalefactor = bottleneck_dim/LR_dim

HRimages = torchvision.datasets.MNIST('/zhome/ab/9/134067/Polytion/AutoEncoder/mnist_setHR/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

idx = HRimages.targets==1
HRimages.data = HRimages.data[idx]

HR_loader = torch.utils.data.DataLoader(HRimages,batch_size=batch_size)#, pin_memory=cuda)

#HR_loader = torch.utils.data.DataLoader(HRimages[:20],batch_size=batch_size) #pin_memory=cuda)
#LR_loader = torch.utils.data.DataLoader(LRimages[:20], batch_size=batch_size) #pin_memory=cuda)

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
lossfunc = torch.nn.MSELoss()
lr = 0.005

num_epochs = 5
def train(model):
    model.train()
    if cuda:
        model = model.cuda()

    epoch_loss = []
    all_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay= 0.001)

    epochs = tqdm.trange(num_epochs, desc="Start training", leave=True)
    try:
        for epoch in epochs:
            batch_loss = []
            for HiResIm, _ in HR_loader:
                HiResIm = HiResIm.float()
                b, c, h, w = HiResIm.size()
                LoResIm = torch.nn.functional.interpolate(HiResIm, scale_factor=0.5).float()

                HiResIm = torch.autograd.Variable(HiResIm).to(device)
                LoResIm = torch.autograd.Variable(LoResIm).to(device)

                output = model(LoResIm).float()
                loss = lossfunc(output, HiResIm).float() /(b*c*h*w)# + lf.TVLoss()(output).float()

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
generatorOptions = {'parallel':True, 'workers':5}
layerOptions = {'randnormweights':True, 'normalize':False, 'parallel':True}

layer = g.PolyclassFTTlayer
model = Autoencoder(layer, layerOptions, generatorOptions)
epoch_loss = train(model)
#epoch_loss = [0, 1, 2, 3]

if epoch_loss[-1] == np.nan or epoch_loss[-1] == np.inf:
    sys.exit()

fig, ax = plt.subplots(1,3, figsize=(15,4))
ax[0].plot(epoch_loss)
ax[0].grid(True)
lossfuncname = str(lossfunc)[0:-2]
ax[0].set_title(lossfuncname + "loss, lr = " + str(lr) + " " + str(layer))
ax[0].set_xlabel('epochs')

model.eval()

test = HRimages.data[0][::2, ::2].reshape(1, 1, 14, 14).to(device)

if cuda:
        output = model(test).reshape(HR_dim,HR_dim).cpu().detach().numpy()
else:
	output = model(test).reshape(HR_dim,HR_dim).detach().numpy()

ax[1].imshow(output, cmap='gray')
ax[1].set_title("Output")
ax[1].axis('off')

ax[2].imshow(HRimages.data[0], cmap='gray')
ax[2].grid(True)
ax[2].set_title("Truth")
ax[2].axis('off')
#torchvision.utils.save_image(output, "outputPolyclassFTTlayer.jpg")

#filename = str(lossfunc)[0:-2] + datetime.now().strftime("%d/%m/%Y_%H_%M_%S") + '.png'
filename = "AutoEncoder/MNIST_" + str(lossfunc)[0:-2] + time.strftime("%d-%m-%Y_%H:%M:%S") + ".png"
plt.savefig(filename, bbox_inches='tight')


