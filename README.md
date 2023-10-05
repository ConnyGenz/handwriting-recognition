# Handwriting-Recognition

## Data
Load the data from kaggle: https://www.kaggle.com/datasets/landlord/handwriting-recognition?select=written_name_train_v2.csv

## Code
There are 7 code files:

read_data.py: reads and preprocesses the data

encode.py: encodes the data

Model.py: Model 

train.py: trains the data and computes the loss

decode.py: decodes the predictions

evaluation.py: evaluates the accuracy of the Model

usage.py: main program file

## Run the Code
In order to run the code, execute the usage.py file adding the directory where you saved the data from kaggle as a command line argument. If you are on the beet server of Heinrich-Heine-Uni, you can use "/data/rafael/", where the data currently lies (October 2023). The most well-trained version of the model is also stored on the beet server under: "/home/cornelia/snap/snapd-desktop-integration/current/Workplace/handwriting-recognition/my_saved_model.pth".

## Toggles
The "Toggles" in the usage.py file allow you to use our code in different ways. Each toggle has an accompanying comment describing the effect of the respective toggle. Before you execute the code, go through the toggles (marked as "##### TOGGLE X) #####") and make sure that you have set each one to the desired setting.

"##### TOGGLE 1) #####": Decide whether to create a model or to use a saved model from a file

"##### TOGGLE 2) #####": Choose whether to train at all or whether to only use the model for inference 

"##### TOGGLE 3) #####": Choose whether to train model from scratch or whether to continue training with a model saved to a file

"##### TOGGLE 4) #####": Choose whether to save the complete trained model to a file and specify filename and path

"##### TOGGLE 5) #####": Choose whether to save just the the parameters of the trained model to a file and specify filename and path

## Variables
In the usage.py file, you will find variables, which you can change according to your wishes: train_size (amount of data used for training), valid_size (amount of data used for validation), test_size (amount of data used for testing), num_epochs (number of epochs the model is supposed to train), mini_batch_size (size of one mini batch).

## Requirements
The python version 3.10.12 was used. The external packages that were used can be found in the "requirements.txt" file.
