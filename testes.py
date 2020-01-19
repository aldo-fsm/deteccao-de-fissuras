#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 08:40:14 2018

@author: aldo
"""

#%%
from matplotlib import pyplot as plt
import numpy as np
import utils

#%%
img = utils.load_image(18)
img = 0.3*img[...,0]+0.59*img[...,1]+0.11*img[...,2]
plt.imshow(img, cmap='gray')
#%%
for i in range(67):
    plt.imshow(utils.load_image(i))
    plt.show()
    print(i)
#%%
plt.imshow(img, cmap='gray')

    
#%%