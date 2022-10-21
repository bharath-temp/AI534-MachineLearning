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
def LR_L2_train(train_data, val_data, lambda_val):
    # Your code here:
    lr = 1
    train_labels = train_data[:,-1]
    train_data = np.delete(train_data, -1, 1)
    val_labels = val_data[:,-1]
    val_data = np.delete(val_data, -1, 1)
    numFeat = np.shape(train_data)[1]
    numDataPoint = np.shape(train_data)[0]
    weights = np.random.rand(numFeat)
    count = 300
    losses = []
    currLossGrad = np.random.rand(numFeat)
    lastLossGrad = np.random.rand(numFeat)

    while count > 0 and LA.norm(currLossGrad) > 1e-5 and LA.norm(currLossGrad) < 1e20:
        # TODO: calculate current loss and gradient
        losses.append(L2_loss(train_data, train_labels, weights, numDataPoint, numFeat,lambda_val))
        currLossGrad = L2_loss_grad(train_data, train_labels, weights, numDataPoint, numFeat, lambda_val)
        if abs(LA.norm(currLossGrad) - LA.norm(lastLossGrad)) < 0.001:
            break
        lastLossGrad = currLossGrad
        weights -= currLossGrad * lr
        count -= 1
        #print(count)
    #plt_loss(losses)
    prediction_training = np.dot(train_data, weights)
    train_acc = accuracy(prediction_training, train_labels)
    prediction_val = np.dot(val_data , weights)
    val_acc = accuracy(prediction_val, val_labels)

    return weights, train_acc, val_acc

def logi(data, weight):
    return 1 / (1 + ma.exp(-np.dot(data, weight)))

def accuracy(prediction, label):
    correct = 0
    for index in range(len(label)):
        if (label[index] == 1 and prediction[index] >= 0):
            correct += 1
        elif (label[index] == 0 and prediction[index] < 0):
            correct += 1
        else:
            correct += 0
    return correct/len(label)

def L2_loss(data, labels, weights, numDataPoint, numFeat, lambda_val):
    loss = 0
    for index in range(numDataPoint):
        loss -= (labels[index] * ma.log(logi(data[index], weights)) + (1-labels[index]) * ma.log(1 - logi(data[index], weights)))
    loss /= numDataPoint
    for feat in range(numFeat):
        loss += weights[feat] * weights[feat] * lambda_val
    return loss
    
def L2_loss_grad(data, labels, weights, numDataPoint, numFeat, lambda_val):
    grad = 0
    for index in range(numDataPoint):
        grad -= (labels[index] - logi(data[index], weights)) * data[index]
    grad /= numDataPoint
    for feat in range(1, numFeat):
        grad[feat] += lambda_val * weights[feat]     
    return grad

# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, lambda_val):
    # Your code here:

    return weights, train_acc, val_acc

# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda_val values and then put multiple loss curves in a single plot.
def plot_losses(accs):
    # Your code here:

    return

def plt_loss(losses):
    xAxis = np.arange(0, len(losses))
    #asked by the assignment to plot y axis in log scale 
    yAxis = losses
    plt.plot(xAxis, yAxis)
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.savefig('lr1e_1lambda1e_1.png')
    plt.close() 
    #plt.show()   
    return losses

# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:

loaded_train_data = load_data("IA2-train.csv")
pre_proc_train_data = preprocess_data(loaded_train_data)
loaded_val_data = load_data("IA2-dev.csv")
pre_proc_val_data = preprocess_data(loaded_val_data)
# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambda_vals
# Your code here:

trained_weights, trained_acc, valed_acc = LR_L2_train(pre_proc_train_data, pre_proc_val_data, 1.99)
print(trained_acc, valed_acc)
# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambda_vals

# Your code here:



