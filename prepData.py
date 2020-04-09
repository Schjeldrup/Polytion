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


images=load_images_from_folder('Images_png_01')

d=512
num=len(images)
img=np.zeros((num,d,d))
for n in range(len(images)):
    img[0]=np.asarray(images[n])
faked = transform.resize(img[0], (img[0].shape[0] / 4, img[0].shape[1] / 4), anti_aliasing=True)

### Normalize
minval= np.min(img[0])
maxval= np.max(img[0])
img2=(img[0]-minval)/maxval
plt.imshow(img2,cmap='gist_gray')
plt.show()
plt.imshow(faked,cmap='gist_gray')
plt.show()
