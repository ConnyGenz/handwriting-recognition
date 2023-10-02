# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:04:07 2022

@author: emibu
"""

from torchmetrics.text import CharErrorRate

# looks at whether the complete name has been predicted correctly, not just some letters
def accuracy_name(decoded_data, identity_train):    
    correct_count = 0
    for k in range(len(decoded_data)):
        if decoded_data[k] == identity_train[k]:
            correct_count += 1
    percentage = correct_count / len(identity_train)
    return correct_count, percentage

# looks at what percentage of letters needs to be changed to go from prediction to groundtruth?
def accuracy_letters(decoded_data, identity_train):
    cer = CharErrorRate()
    cer_value = cer(decoded_data, identity_train)
    return cer_value
