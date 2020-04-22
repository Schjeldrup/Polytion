# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:18:30 2020

@author: Martin Jessen
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import transform
from PIL import Image

def load_images_from_folder(folder):
    images=[]
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        images.append(np.asarray(img))
    return images

def load_images_from_all_folders(folder):
    images = []
    for root, dirs, files in os.walk(folder, topdown=True):        
        for file in files:
            img = Image.open(os.path.join(root, file))
            images.append(np.asarray(img))
            img.close()
            
#    for filename in os.listdir(folder):
#        img = Image.open(os.path.join(folder,filename))
#        images.append(img)
    return images

def normalize(images):
    ### Normalize
    minval= 29744
    maxval= 33000
    for i in range(len(images)):
        images[i]=(images[i]-minval)/maxval
    return images

def compress_images(images):
    img=[]
    for n in range(len(images)):
        img.append(transform.resize(images[0], (images[0].shape[0] / 4, images[0].shape[1] / 4), anti_aliasing=True))
    return img

def show_img(image):
    plt.imshow(image,cmap='gist_gray')
    plt.show()
