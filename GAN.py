# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import PolytionLayers as PL
import prepData as prep
from torch.utils.data import DataLoader
import numpy as np  
import skimage.transform

"""
Loader
"""

#images=prep.load_images_from_folder('Images_png_01/Images_png/000001_01_01')
images=prep.load_images_from_folder('/work3/projects/s181603-Jun-2020/Images_png/000047_08_01')

lowresimages=prep.compress_images(images)

"""
downscaler
"""

batch_size = 1
lowres_loader = DataLoader(lowresimages, batch_size=batch_size)#, pin_memory=cuda)
real_data_loader = DataLoader(images,batch_size=batch_size)

lr_dim=128
hr_dim=512

"""
GAN Structure
"""
"""
Generator
"""
# Save the desired order and rank of the following algorithms here:
N = 5
rank = 4
scalefactor = 4

# Save a standard set of inputs
imwidth, imheight = 512, 512
my_generator = PL.Generator(PL.PolyGAN_CP_Layer, N, rank, imwidth, imheight, scalefactor)


"""
Discriminator   
"""

#Takes a low-resolution pic and a high-resolution and outputs a probability of it being a fake
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        # On the {z_s,d}
        self.disc = nn.Sequential(
        nn.Conv2d(1, 16 , kernel_size=3, stride=1,padding=1),
        nn.ReLU()
        )
        
        self.disc2 = nn.Sequential(
        nn.Conv2d(2, 2 , kernel_size=3, stride=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(2, 4 , kernel_size=1, stride=1,padding=0),
        nn.ReLU(),

        nn.Conv2d(4, 4 , kernel_size=3, stride=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(4, 8 , kernel_size=3, stride=1,padding=0),
        nn.ReLU(),
        
        nn.Conv2d(8, 8 , kernel_size=3, stride=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(8, 1 , kernel_size=3, stride=1,padding=0),
        nn.ReLU()
        )
        
        self.disc3 = nn.Sequential(
        nn.Linear(in_features=252004, out_features=1, bias=True),
        nn.Sigmoid()
        )
        
        
    def forward(self, lowres,highres):
        upscaledlow=self.disc(lowres).type('torch.DoubleTensor')
        upscaledlow=upscaledlow.view(batch_size,1,512,512)
        combine= torch.cat([upscaledlow,highres.type('torch.DoubleTensor')], dim=1).type('torch.DoubleTensor')
        res=self.disc2(combine)
        res=res.view(batch_size,252004)
        res=self.disc3(res)
        return res 
        

my_discriminator = discriminator().type('torch.DoubleTensor')

print(my_generator.parameters)
print(my_discriminator.parameters)

"""    
gpu_generator = generator().to(device)
gpu_discriminator = discriminator().to(device)
"""

#Declare Loss
loss = nn.BCELoss() #We should try other loss-functions

#Learning rate
generator_optim = torch.optim.Adam(my_generator.parameters(), 2e-4, betas=(0.5, 0.9))
discriminator_optim = torch.optim.Adam(my_discriminator.parameters(), 2e-4, betas=(0.5, 0.9))

lr_scheduler = np.linspace(0,2e-4,30)


#Prepare Training Loop
discriminator_loss = []
generator_loss = []

#Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    """
    if epoch in range(30):
        generator_optim = torch.optim.Adam(my_generator.parameters(), lr_scheduler[epoch], betas=(0.5, 0.9))
        discriminator_optim = torch.optim.Adam(my_discriminator.parameters(), lr_scheduler[epoch], betas=(0.5, 0.9))
    """
    for i, data in enumerate(zip(real_data_loader,lowres_loader)):
        (realhighres, lowres) = data
        true_label = torch.ones(batch_size, 1).type('torch.DoubleTensor')#.to(device) 
        fake_label = torch.zeros(batch_size, 1).type('torch.DoubleTensor')#.to(device)
        my_discriminator.zero_grad()
        my_generator.zero_grad()
        
        #define data
        realhighres=realhighres.view(batch_size,1,512,512)#!!!
        x_true = Variable(realhighres)#!!!.to(device)
        lowres=lowres.view(batch_size,1,128,128)
        lowres = Variable(lowres)#!!!
        
        
        print('before discriminator number: {}'.format(i))
        output = my_discriminator(lowres,x_true)

        error_true = loss(output, true_label)
        error_true.backward()
        print('before gen: {}'.format(i))
        # Step 2. Generate fake data G(z) 
        x_fake = my_generator(lowres.type('torch.FloatTensor'))
        x_fake = Variable(x_fake)
        # Step 3. Send fake data through discriminator
        #         propagate error and update D weights.
        # --------------------------------------------
        # Note: detach() is used to avoid updating generator gradients
        
        output = my_discriminator(lowres,x_fake.detach()) 
        error_fake = loss(output, fake_label)
        error_fake.backward()
        discriminator_optim.step()
        print('before fake discriminator number: {}'.format(i))
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = my_discriminator(lowres,x_fake)
        error_generator = loss(output, true_label)
        error_generator.backward()
        generator_optim.step()
        batch_d_loss.append((error_true/(error_true + error_fake)).item())
        batch_g_loss.append(error_generator.item())
        
        
    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    

