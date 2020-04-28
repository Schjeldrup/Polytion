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
        #one from each folder
        try:
            img = Image.open(os.path.join(root, files[0]))
            if np.size(np.asarray(img))==262144:
                images.append(np.asarray(img))
            img.close()
        except:
            print('Unable to open file')
        """
        for file in files:
            img = Image.open(os.path.join(root, file))
            images.append(np.asarray(img))
            img.close()
        """
#    for filename in os.listdir(folder):
#        img = Image.open(os.path.join(folder,filename))
#        images.append(img)
    print('there are {} images'.format(len(images)))
    sizes=[]
    for q in range(len(images)):
        sizes.append(np.size(images[q]))
    print(sizes)
    return images

def normalize(images):
    ### Normalize
    minval= 29744
    maxval= 33000
    for i in range(len(images)):
        images[i]=(images[i]-minval)/maxval
    return images

def normalize_0(images):
    ### Normalize to 0-1
    normalized = []
    for img in images:
        lmin = np.float(np.min(img))
        lmax = np.float(np.max(img))
        normalized.append( (img - lmin)/(lmax-lmin) )
    return normalized

def normalize_32768(images):
    ### Normalize to 0-1
    for img in images:
        img = img - 32768
    return images

def compress_images(images):
    img=[]
    for n in range(len(images)):
        img.append(transform.resize(images[n], (images[n].shape[0] / 4, images[n].shape[1] / 4), anti_aliasing=True))
    return img

def show_img(image):
    plt.imshow(image,cmap='gist_gray')
    plt.show()
