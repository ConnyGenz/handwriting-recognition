# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:13:46 2022

@author: emibu
"""
import numpy as np
import torch

# find the length of longest name in data (purpose: other labels need to be padded to that length)
def max_str(train, train_size):
    liste1 = [train.loc[i,"IDENTITY"] for i in range(train_size)]
    all_lists = liste1
    longest_name = max(all_lists, key=len)
    return len(longest_name)
        

def min_str(train, train_size):
    liste1 = [train.loc[i,"IDENTITY"] for i in range(train_size)]
    all_lists = liste1
    longest_name = min(all_lists, key=len)
    return len(longest_name)

# convert one name (label) to an array of numbers as definded in the alphabets dictionary
def label_to_num(label, alphabets):
    label = list(label)
    list = [alphabets[i] for i in label]
    return np.array(list)
   

# encode all ground-truth strings (reference) from input data and return tensor
def encode_labels(size, data, max_str_len, alphabets, device):
    # placeholder for real labels
    label_placeholder = np.ones([size, max_str_len]) * 0      
    # placeholder is filled with real data in encoded form
    for i in range(size):
        # -1 remains if label is shorter than max_str_len
        label_placeholder[i, 0:len(data.loc[i, 'IDENTITY'])] = label_to_num(data.loc[i, 'IDENTITY'],alphabets) 
    # convert labels to torch tensor
    encoded_labels = torch.tensor(label_placeholder, dtype=torch.float32).to(device)
    return encoded_labels
     