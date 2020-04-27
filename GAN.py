# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np  
import skimage.transform

import Discriminator as disc
import Generator as gen
import prepData as prep

"""
Parameters
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Compression Parameters
N = 5
rank = 4
scalefactor = 4

#Define size
imwidth, imheight = 512, 512

#Training parameters
loss = nn.BCELoss() #We should try other loss-functions
num_epochs = 1
batch_size = 13

"""
Loader
"""

#images=prep.load_images_from_folder('Images_png_01/Images_png/000001_01_01')
images=prep.load_images_from_folder('000001_01_01')
#images=prep.load_images_from_folder('/work3/projects/s181603-Jun-2020/Images_png/000001_01_01/')

#images=prep.load_images_from_all_folders('/work3/projects/s181603-Jun-2020/Images_png/')

images=prep.normalize(images)
images[0] = torch.rand(imwidth, imheight).double().numpy()
lowresimages=prep.compress_images(images)

lowres_loader = DataLoader(lowresimages, batch_size=batch_size)#, pin_memory=cuda)
real_data_loader = DataLoader(images,batch_size=batch_size)

prep.show_img(lowresimages[0])

"""
GAN Structure
"""
"""
Generator
"""
#Imports generator from file
my_generator = gen.Generator(gen.FTT_Layer, N, rank, imwidth, imheight,scalefactor)


"""
Discriminator   
"""
#Imports discriminator from file   
my_discriminator = disc.discriminator(batch_size).type('torch.DoubleTensor')

"""    
gpu_generator = generator().to(device)
gpu_discriminator = discriminator().to(device)
"""



#Learning rate
generator_optim = torch.optim.Adam(my_generator.parameters(), 2e-2)#, betas=(0.5, 0.9))
discriminator_optim = torch.optim.Adam(my_discriminator.parameters(), 2e-5)#, betas=(0.5, 0.9))

lr_scheduler = np.linspace(0,2e-4,30)


#Prepare Training Loop
discriminator_true_loss = []
discriminator_fake_loss = []
generator_loss = []

#Training Loop

for epoch in range(num_epochs):
    print('Epoch number: {}'.format(epoch))
    batch_d_loss_true, batch_d_loss_fake, batch_g_loss = [], [], []
    """
    if epoch in range(30):
        generator_optim = torch.optim.Adam(my_generator.parameters(), lr_scheduler[epoch], betas=(0.5, 0.9))
        discriminator_optim = torch.optim.Adam(my_discriminator.parameters(), lr_scheduler[epoch], betas=(0.5, 0.9))
    """
    for i, data in enumerate(zip(real_data_loader,lowres_loader)):
        (realhighres, lowres) = data
        batch_size=realhighres.size(0)
        true_label = torch.ones(batch_size, 1).type('torch.DoubleTensor')-0.001#.to(device) 
        fake_label = torch.zeros(batch_size, 1).type('torch.DoubleTensor')+0.001#.to(device)

        my_discriminator.zero_grad()
        my_generator.zero_grad()

        #define data
        realhighres=realhighres.view(batch_size,1,512,512)
        x_true = Variable(realhighres.type('torch.DoubleTensor'))#!!!.to(device)
        lowres=lowres.view(batch_size,1,128,128)
        lowres = Variable(lowres.type('torch.DoubleTensor'))#!!!
        
        
        print('before discriminator number: {}'.format(i))
        output = my_discriminator(lowres,x_true)
        print(output)
        #print(output[0]-output[1])
        print('true answer: {}'.format(true_label[0]))
        error_true = loss(output, true_label)
        error_true.backward()
        print('training discriminator number: {}'.format(i))
        # Step 2. Generate fake data G(z) 
        x_fake = my_generator(lowres.type('torch.FloatTensor'))
        x_fake = Variable(x_fake).type('torch.DoubleTensor')
        # Step 3. Send fake data through discriminator
        #         propagate error and update D weights.
        # --------------------------------------------
        # Note: detach() is used to avoid updating generator gradients
        output = my_discriminator(lowres,x_fake.detach()) 
        print(output)
        #print(output[0]-output[1])
        print('true answer: {}'.format(fake_label[0]))
        error_fake = loss(output, fake_label)
        error_fake.backward()
        discriminator_optim.step()
        
        print('Training generator number: {}'.format(i))
        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = my_discriminator(lowres,x_fake)
        print(output)
        #print(output[0]-output[1])
        print('true answer: {}'.format(true_label[0]))
        error_generator = loss(output, true_label)
        error_generator.backward()
        generator_optim.step()
        batch_d_loss_true.append((error_true).item())
        batch_d_loss_fake.append((error_fake).item())
        batch_g_loss.append(error_generator.item())
        
    
        
    discriminator_true_loss.append(np.mean(batch_d_loss_true))
    discriminator_fake_loss.append(np.mean(batch_d_loss_fake))
    generator_loss.append(np.mean(batch_g_loss))
    

# -- Plotting --
f, axarr = plt.subplots(1, 2, figsize=(15, 5))

# Loss
ax = axarr[0]
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
    
ax.plot(np.arange(epoch+1), discriminator_true_loss)
ax.plot(np.arange(epoch+1), discriminator_fake_loss)
ax.plot(np.arange(epoch+1), generator_loss, linestyle="--")
ax.legend(['Discriminator', 'Generator'])
    
print('exited')
print(discriminator_true_loss)
print(discriminator_fake_loss)
print(generator_loss)

prep.show_img(my_generator(lowres.type('torch.FloatTensor')).detach().numpy()[0][0])


