#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import time
from tqdm import tqdm

import Discriminator as d
import Generator as g
import prepData as prep


# Parameters:
batch_size = 13
N = 5
rank = 15

LR_dim = 128
HR_dim = 512
bottleneck_dim = 32

scalefactor = HR_dim/bottleneck_dim
downscalefactor = bottleneck_dim/LR_dim


images = prep.load_images_from_folder('000001_01_01')
HRimages = prep.normalize(images)
LRimages = prep.compress_images(HRimages)



HR_loader = DataLoader(HRimages[:20],batch_size=batch_size)#, pin_memory=cuda)
LR_loader = DataLoader(LRimages[:20], batch_size=batch_size)#, pin_memory=cuda)



# In[ ]:


class Autoencoder(torch.nn.Module):
    def __init__(self, layer, layerOptions, generatorOptions):
        super(Autoencoder,self).__init__()
        self.encoder = g.Generator(layer, N, rank, bottleneck_dim, bottleneck_dim, downscalefactor, layerOptions, generatorOptions)
        self.decoder = g.Generator(layer, N, rank, HR_dim, HR_dim, scalefactor, layerOptions, generatorOptions)

    def forward(self, x):
        #print("Input shape = ", x.shape)
        #print(x.sum())
        x = self.encoder(x.float())
        #print("After encoder shape = ", x.shape)
        #print(x.sum())
        x = self.decoder(x)
        #print("After decoder shape = ", x.shape)
        #print(x.sum())
        return x

loss_func = torch.nn.MSELoss()


# In[ ]:


num_epochs = 20
batch_size = 5

def train():
    model.train()
    epoch_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm(range(num_epochs)):
        batch_loss = []
        for HiResIm, LoResIm in zip(HR_loader, LR_loader):
            HiResIm = Variable(HiResIm.unsqueeze_(1).float())
            LoResIm = LoResIm.unsqueeze_(1).float()

            output = model(LoResIm).float()

            loss = loss_func(output, HiResIm).float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(np.mean(batch_loss))
    return epoch_loss


# ## 3. Training the different layers and generators:

# In[ ]:


generatorOptions = {'parallel':True, 'workers':3}
layerOptions = {'randnormweights':True, 'normalize':False, 'parallel':False}

print("Start training")

#PolyganCPlayer:
model = Autoencoder(g.PolyganCPlayer, layerOptions, generatorOptions)
epoch_loss = train()
plt.plot(epoch_loss)
plt.savefig('PolyganCPlayer.png', bbox_inches='tight')

model.eval()
test = torch.tensor(LRimages[0]).reshape(1,1,LR_dim,LR_dim)
output = model(test).reshape(HR_dim,HR_dim)
torchvision.utils.save_image(output, "testoutput.jpg")

# PolyclassFTTlayer:
model = Autoencoder(g.PolyclassFTTlayer, layerOptions, generatorOptions)
epoch_loss = train()
plt.plot(epoch_loss)
plt.savefig('PolyclassFTTlayer.png', bbox_inches='tight')

model.eval()
test = torch.tensor(LRimages[0]).reshape(1,1,LR_dim,LR_dim)
output = model(test).reshape(HR_dim,HR_dim)
torchvision.utils.save_image(output, "testoutput.jpg")

print("done")
