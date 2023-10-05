# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:23:07 2022

@author: emibu
"""

# Import libraries
import os
from pathlib import Path
import torch
import argparse

# import from own project
from read_data import read_labels
from read_data import encode
from encode import max_str
from encode import min_str
from encode import encode_labels
from Model import CharModel
from train import create_mini_batches
from train import train_loss
from decode import decode_preds
from decode import ctc_decode
from evaluation import accuracy_name
from evaluation import accuracy_letters


#%% Variables

train_size = 1088
valid_size = 30
test_size = 10
num_epochs = 10

mini_batch_size = 25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

parser = argparse.ArgumentParser()
parser.add_argument("command_line_path", help='takes the path to the directory where the data to be processed by the model is stored')
args = parser.parse_args()
user_path = args.command_line_path

# character to number
alphabets = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,
           "L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,
           "V":22,"W":23,"X":24,"Y":25,"Z":26,"-":27,"'":28," ":29}     


num_of_characters = len(alphabets) + 1  # +1 for ctc pseudo blank
num_of_timesteps = 64                   # length of predicted labels (for images to be divided into 64 time steps); arbitrary     

#%% Preprocessing

path = Path(user_path)
#path = args.command_line_path
os.chdir(path)
train_data = read_labels("written_name_train_v2.csv")
# valid_data = read_labels("written_name_validation_v2.csv")
test_data = read_labels("written_name_test_v2.csv")

# use encode function from "read_data" file
train_x_new = encode(user_path, "train", train_size, train_data, device)
# valid_x_new = encode("validation", valid_size, valid_data, device)
test_x_new = encode(user_path, "test", test_size, test_data, device)


#%% Variables #2

max_str_len = max_str(train_data, train_size)
# Target sequence length of longest target in batch (padding length)
min_str_len = min_str(train_data, train_size)
# Minimum target length

#%% Encode

train_y = encode_labels(train_size, train_data, max_str_len, alphabets, device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)

test_y = encode_labels(test_size, test_data, max_str_len, alphabets, device) 
test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

#%% Model

##### TOGGLE 1) #####
# Decide whether to create a model or to use a saved model from a file
work_with_model_from_file = False

if work_with_model_from_file:
    print("Loading model parameters from file")
    location_of_parameters = "/home/cornelia/snap/snapd-desktop-integration/current/Workplace/handwriting-recognition/my_saved_model.pth"
    cm = CharModel(29)
    cm.load_state_dict(torch.load(location_of_parameters))
    cm.eval()
    cm.to(device)

if not work_with_model_from_file:
    cm = CharModel(29).to(device) #29 characters in alphabets


#%% Training

##### TOGGLE 2) #####
# Choose whether to train at all
do_train = True

##### TOGGLE 3) #####
# Choose whether to train model from scratch or whether to continue training with a model saved to a file
train_with_model_from_file = False

if do_train:
    train_loss(num_of_timesteps, train_size, mini_batch_size, train_x_new,
               max_str_len, train_y, cm, num_epochs, train_data, device)
    
if train_with_model_from_file:

    print("Loading model from file ...")
    cm = torch.load("/home/cornelia/snap/snapd-desktop-integration/current/Workplace/handwriting-recognition/my_saved_model.pth")
    cm.train()

    train_loss(num_of_timesteps, train_size, mini_batch_size, train_x_new, max_str_len, train_y, cm, num_epochs, train_data, device)


##### TOGGLE 4) #####
# Choose whether to save the complete trained model to a file and specify filename and path
# PyTorch Tutorial 17 - Saving and Loading Models: https://www.youtube.com/watch?v=9L9jEOwRrCg 
save_trained_model = False

##### TOGGLE 5) #####
# Choose whether to save just the the parameters of the trained model to a file and specify filename and path
# PyTorch Tutorial 17 - Saving and Loading Models: https://www.youtube.com/watch?v=9L9jEOwRrCg 
save_model_parameters = False

if save_trained_model: 
    save_under = "/home/cornelia/snap/snapd-desktop-integration/current/Workplace/handwriting-recognition/new_saved_model.pth"
    print("Saving complete model to file " + str(save_under))
    torch.save(cm, save_under)

if save_model_parameters:
    storage_location = "/home/cornelia/snap/snapd-desktop-integration/current/Workplace/handwriting-recognition/new_saved_model.pth"
    print("Saving model parameters to file " + str(storage_location))
    torch.save(cm.state_dict(), storage_location)


#%% Decode model output and take a look at results (with training data)
# Was used to manually assess performance of model during training 

# Create predictions for mini-batches:
mini_x_for_pred = create_mini_batches(train_x_new, mini_batch_size)

train_data_permuted_batch_zero = torch.permute(mini_x_for_pred[0], (0,3,2,1))
predictions_for_batch_zero = cm(train_data_permuted_batch_zero)  #use the CharModel

train_data_permuted_batch_five = torch.permute(mini_x_for_pred[5], (0,3,2,1))
predictions_for_batch_five = cm(train_data_permuted_batch_five)  #use the CharModel

encoded_zero = decode_preds(predictions_for_batch_zero, mini_batch_size, alphabets)
encoded_five = decode_preds(predictions_for_batch_five, mini_batch_size, alphabets)
# >> result is a list of strings of the form "AAAA°NNN°NNNNNN°AA" (name "Anna", uncleaned)

# >> Derive “Anna” from “"AAAA°NNN°NNNNNN°AA"
decoded_zero = ctc_decode(encoded_zero)
decoded_five = ctc_decode(encoded_five)

# print("decoded names from batch 0: ", decoded_zero[0:24])
# print("decoded names from batch 5: ", decoded_five[0:24])


#%% Evaluation - check accuracy and error rate on test set

test_data_permuted = torch.permute(test_x_new, (0,3,2,1))
predictions_for_test_data = cm(test_data_permuted)  #use the CharModel
encoded_test_predictions = decode_preds(predictions_for_test_data, test_size, alphabets)
decoded_test_predictions = ctc_decode(encoded_test_predictions)

print("Names predicted by the model for the data from the test set: \n")
print(decoded_test_predictions)

complete_list_of_correct_names = test_data['IDENTITY'].tolist()
list_of_correct_names_test_size = complete_list_of_correct_names[0:test_size]

number_of_correct_names, percentage = accuracy_name(decoded_test_predictions, list_of_correct_names_test_size)
number_of_wrong_characters = accuracy_letters(decoded_test_predictions, list_of_correct_names_test_size) 

print("\n The number of correct names in the test set of size " + str(test_size) + " is: " + str(number_of_correct_names))
print("\n The percentage of correct names in the test set of size " + str(test_size) + " is: " + str(percentage))
print("\n The percentage of wrong letters in the total number of " + str(test_size) + " letters is: " + str(number_of_wrong_characters))

