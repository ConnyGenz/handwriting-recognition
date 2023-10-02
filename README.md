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

### Run the Code
In order to run the code, download the data from kaggle. In the usage.py file, under "#%% Preprocessing", change the path to the directory where you saved the data. Insert that path also in the read_data.py file in the encode function for the "beginning_of_path" variable. In order to run the code, execute the usage.py file.

## Toggles

The "Toggles" in the usage.py file allow you to use our code in different ways. Each toggle has an accompanying comment describing the effect of the respective toggle. Before you execute the code, go through the toggles (marked as "##### TOGGLE X) #####") and makes sure that you have set each one to the desired setting.

"##### TOGGLE 1) #####"
Decide whether to create a model or to use a saved model from a file

"##### TOGGLE 2) #####"
Choose whether to train at all

"##### TOGGLE 3) #####"
Choose whether to train model from scratch or whether to continue training with a model saved to a file

"##### TOGGLE 4) #####"
Choose whether to save the complete trained model to a file and specify filename and path

"##### TOGGLE 5) #####"
Choose whether to save just the the parameters of the trained model to a file and specify filename and path

## Requirements
The python version 3.10.12 was used. The external packages that were used can be found in the "requirements.txt" file.
