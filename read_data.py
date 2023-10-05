# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:05:47 2022

@author: emibu
"""

import pandas as pd
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
# following two imports are optional, used for displaying images and plotting data
#from matplotlib import image as img
#from matplotlib import pyplot

# generate images of shape height=64 and width=256
# from kaggle: https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/notebook#Check-model-performance-on-validation-set
def preprocess(img):
    (h, w) = img.shape
    final_img = np.ones([64, 256])*255 # 255 = blank white image
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
        
    final_img[:h, :w] = img  # size 64x256
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


# encode images to input tensors
# from kaggle: https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/notebook#Check-model-performance-on-validation-set
def encode_image(beginning_of_path, given_path, size, data, device):
    data_x = []
    combined_path = beginning_of_path + given_path
    os.chdir(Path(combined_path))

    for i in range(size):
        image = np.array(Image.open(data.loc[i,'FILENAME']).convert('L'))
        image = preprocess(image)
        image = image/255.
        data_x.append(image)
        
    data_x = np.array(data_x).reshape(-1, 256, 64, 1)
    data_x = torch.tensor(data_x, dtype=torch.float32).to(device)
    return data_x


def read_labels(file):
    data = pd.read_csv(file)
    data = data.dropna() # drop NaN values
    data = data[data["IDENTITY"]!="UNREADABLE"]   # exclude UNREADABLE labels
    data["IDENTITY"] = data["IDENTITY"].str.upper()     # all upper case
    data.reset_index(inplace = True, drop=True)     # reset index
    
    return data


'''
#---------------------------- view data      
path = "where/you/saved/the/images"
os.chdir(path)        
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('TEST_0001.jpg') 
imgplot = plt.imshow(img)
plt.show()
'''        

  
    