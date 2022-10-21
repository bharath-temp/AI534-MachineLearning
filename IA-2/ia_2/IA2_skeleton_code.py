# AI534
# IA2 skeleton code
# By: Bharath, Cheng, Bhargav

import numpy as np
import pandas as pd
import matplotlib


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    loaded_data = pd.read_csv(path)
    return loaded_data

# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.

def preprocess_data(data):
    # Your code here:
    data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
    data['Annual_Premium'] = (data['Annual_Premium'] - data['Annual_Premium'].min()) / (data['Annual_Premium'].max() - data['Annual_Premium'].min())
    data['Vintage'] = (data['Vintage'] - data['Vintage'].min()) / (data['Vintage'].max() - data['Vintage'].min())
    preprocessed_data = data.to_numpy()

    return preprocessed_data

# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L2_train(train_data, val_data, lambda_val):
    # Your code here:

    return weights, train_acc, val_acc

# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, lambda_val):
    # Your code here:

    return weights, train_acc, val_acc

# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_losses(accs):
    # Your code here:

    return

# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:


# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambdas
# Your code here:


# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Your code here:



loaded_data = load_data("IA2-train.csv")
pre_proc_data = preprocess_data(loaded_data)





