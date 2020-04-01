# -*- coding: utf-8 -*-
"""
import torch.nn as nn
import numpy as np
import torch
import tntorch as tn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import Polytion.py
import prepData.py
torch.set_default_dtype(torch.float64)
from torch.utils.data import DataLoader


"""
Loader
"""

images=load_images_from_folder('Images_png_01')

"""
downscaler
"""

batch_size = 1
lowres_loader = DataLoader(images, batch_size=batch_size)#, pin_memory=cuda)
real_data_loader = DataLoader(images,batch_size=batch_size)

"""
GAN Structure
"""


  
        
        

#Takes a low-resolution pic and a high-resolution and outputs a probability of it being a fake
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        # On the {z_s,d}
        self.disc = nn.Sequential(
        nn.Conv2d(x, x, kernel_size=3, stride=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(x, y, kernel_size=4, stride=2,padding=0),
        nn.ReLU(),
        
        nn.Conv2d(y, y, kernel_size=3, stride=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(y, z, kernel_size=4, stride=2,padding=0),
        nn.ReLU(),
        
        nn.Conv2d(z, z, kernel_size=3, stride=1,padding=0),
        nn.ReLU(),
        nn.Conv2d(z, q, kernel_size=4, stride=2,padding=0),
        nn.ReLU(),
        
        nn.Conv2d(q, q, kernel_size=3, stride=1,padding=0),
        nn.ReLU())

    def forward(self, lowres,highres):
        combine= torch.cat([lowres,highres], dim=1)
        return self.disc(combine)
        
my_generator = generator()
my_discriminator = discriminator()

print(my_generator.parameters)


from torch.autograd import Variable
loss = nn.BCELoss()  

"""    
gpu_generator = generator().to(device)
gpu_discriminator = discriminator().to(device)
"""

#Declare Parameters
loss = nn.BCELoss()
batch_size = 1

#Learning rate
generator_optim = torch.optim.Adam(generator(), 2e-4, betas=(0.5, 0.9))
discriminator_optim = torch.optim.Adam(discriminator(), 2e-4, betas=(0.5, 0.9))

lr_scheduler = np.linspace(0,2e-4,30)


#Prepare Training Loop
discriminator_loss = []
generator_loss = []

#Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    if epoch in range(30):
        generator_optim = torch.optim.Adam(my_generator.parameters(), lr_scheduler[epoch], betas=(0.5, 0.9))
        discriminator_optim = torch.optim.Adam(my_discriminator.parameters(), lr_scheduler[epoch], betas=(0.5, 0.9))
    for i, data in enumerate(zip(real_data_loader,attribute_loader)):
        (split, into, parts) = data
        true_label = torch.ones(batch_size, 1)#.to(device)
        fake_label = torch.zeros(batch_size, 1)#.to(device)
        discriminator.zero_grad()
        generator.zero_grad()
        
        # Step 1. Send real data through discriminator and backpropagate its errors.
        x_true = Variable(real_data)#.to(device)
        output = discriminator(x_true)
        
        error_true = loss(output, true_label)
        error_true.backward()
        
        # Step 2. Generate fake data G(z)
        noise = torch.randn(batch_size, 100, 1,1)
        noise = Variable(noise, requires_grad=False).to(device)
        
        x_fake = generator(var)
        
        # Step 3. Send fake data through discriminator
        #         propagate error and update D weights.
        # --------------------------------------------
        # Note: detach() is used to avoid updating generator gradients
        output = discriminator(x_fake.detach(),downsampled,attributes) 
        
        error_fake = loss(output, fake_label)
        error_fake.backward()
        discriminator_optim.step()
        
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = discriminator(x_fake)
        
        error_generator = loss(output, true_label)
        error_generator.backward()
        generator_optim.step()
        
        batch_d_loss.append((error_true/(error_true + error_fake)).item())
        batch_g_loss.append(error_generator.item())
        
        
    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))
    
plt.scatter()
    