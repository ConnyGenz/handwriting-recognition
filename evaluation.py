# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:04:07 2022

@author: emibu
"""

from torchmetrics.text import CharErrorRate

# looks at whether the complete name has been predicted correctly, not just some letters
def accuracy_name(prediction, identity):    
    correct_count = 0
    for k in range(len(prediction)):
        if prediction[k] == identity[k]:
            correct_count += 1
    percentage = correct_count / len(identity)
    return correct_count, percentage

# looks at how many characters need to be changed to go from prediction to ground truth
def accuracy_letters(prediction, identity):
    cer = CharErrorRate()
    cer_value = cer(prediction, identity)
    return cer_value
