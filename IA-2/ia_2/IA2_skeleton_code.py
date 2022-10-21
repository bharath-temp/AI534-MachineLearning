# AI534
# IA2 skeleton code

import math as ma
from ast import Num
import numpy as np
import pandas as pd
import matplotlib
from numpy import linalg as LA
import matplotlib.pyplot as plt

# By: Bharath, Cheng, Bhargav




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
def LR_L2_train(train_data, val_data, lambda):
    # Your code here:
    lr = 0.01
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    weights = []
    numFeat = np.shape(train_data)[1]
    numDataPoint = 0
    count = 1000
    losses = []
    currLossGrad = np.random.rand(numFeat)
    lastLossGrad = np.random.rand(numFeat)

    while count > 0 and LA.norm(currLossGrad) > 1e-5 and LA.norm(currLossGrad) < 1e20:
        # TODO: calculate current loss and gradient
        losses.append(L2_loss(data, labels, weights, numDataPoint, numFeat,lambda))
        currLossGrad = L2_loss_grad(data, labels, weights, numDataPoint, numFeat, lambda)
        if abs(LA.norm(currLossGrad) - LA.norm(lastLossGrad)) < 0.001:
            break
        lastLossGrad = currLossGrad
        weights -= currLossGrad * lr
        count -= 1
        print(count)
    
    prediction_training = train_data * weights
    train_acc = accuracy(prediction_training, train_labels)
    prediction_val = val_data * weights
    val_acc = accuracy(prediction_val, val_labels)

    return weights, train_acc, val_acc

def logi(data, weight):
    return 1 / (1 + ma.exp(-np.dot(data, weight)))

def accuracy(prediction, label):
    correct = 0
    for index in range(len(label)):
        if label == 1 && prediction >= 0:
            correct += 1
        else if label == 0 && prediction < 0:
            correct += 1
    return correct/len(label)

def L2_loss(data, labels, weights, numDataPoint, numFeat, lambda):
    loss = 0
    for index in range(numDataPoint):
        loss -= (labels[index] * ma.log(logi(data[index], weights[index])) + (1-labels[index]) * ma.log(1 - logi(data[index], weights[index])))
    loss /= numDataPoint
    for feat in range(numFeat):
        loss += weights[feat] * weights[feat] * lambda
    return loss
    
def L2_loss_grad(data, labels, weights, numDataPoint, numFeat, lambda):
    grad = 0
    for index in range(numDataPoint):
        grad -= (labels[index] - logi(data[index], weights[index])) * data[index]
    grad /= numDataPoint
    for feat in range(1, numFeat):
        grad[feat] += lambda * weights[feat]     
    return grad

# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, lambda):
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