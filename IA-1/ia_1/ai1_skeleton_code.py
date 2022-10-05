# CS 534
# AI1 skeleton code
# By Quintin Pope

from ast import Num
import numpy as np
import pandas as pd
import matplotlib



# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    #loaded_data = np.genfromtxt(path, delimiter=',')
    df_from_csv = pd.read_csv(path)
    print(df_from_csv)
    return df_from_csv

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    # Your code here:
    # Split date to YYMMDD
    data[["month", "day", "year"]] = data["date"].str.split("/", expand=True)
    data['year'] = data['year'].astype(int)
    data['month'] = data['month'].astype(int)
    data['day'] = data['day'].astype(int)
    #print(pd.DataFrame(data))
    preprocessed_data = pd.DataFrame(data).to_numpy()
    # Delete the column of house ID, and the column of date
    preprocessed_data = np.delete(preprocessed_data, [0,1], 1)
    
    # Add a constant "1" column
    numRow = np.shape(preprocessed_data)[0]
    preprocessed_data = np.c_[preprocessed_data, np.ones(numRow)]
    #  yr renovated to "age since built"
    for index in range(numRow):
        if preprocessed_data[index][12] == 0:
            preprocessed_data[index][12] = preprocessed_data[index][-2] - preprocessed_data[index][11]
        else:
            preprocessed_data[index][12] = preprocessed_data[index][-2] - preprocessed_data[index][12] 
    # remove redundant livingsqft if being asked
    if drop_sqrt_living15 == 1:
        preprocessed_data = np.delete(preprocessed_data, 16, 1)
    # Calculate mean value by col
    meanVal = np.mean(preprocessed_data, dtype=np.float64, axis=0)
    # Calculate std value by col
    stdVal = np.std(preprocessed_data, dtype=np.float64, axis=0)
    # Normalize each column by substracting to mean before diving by std
    for col in range(23):
        # Do not implement norm on waterfront, price, and constant 1
        if col != 5 and col != 18 and col!= 22:
            preprocessed_data[:,col] -= meanVal[col]
            preprocessed_data[:,col] /= stdVal[col]

    return preprocessed_data




# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr):
    # Your code here:
    # Number of features = number of columns from pre-processed data - 1 : this is because data contains one column of label
    numFeat = np.shape(data)[1] - 1
    # Number of training data = number of rows from pre-processed data \
    numDataPoint = np.shape(data)[0] 
    # Initialize weights to all zero 
    weights = np.zeros(numFeat)
    # this array stores the loss after each iteration
    losses = []
    #we are supposed to run 4000 iterations if the loss is not convereged 
    count = 4000
    # this record loss gradient for the current iteration
    currLossGrad = 10
    # termination conditions
 
    while count > 0 and currLossGrad > 1:
        # TODO: calculate current loss and gradient
        losses.append(loss(data, labels, weights))
        currLossGrad = lossGrad(data, labels, weights, numDataPoint, numFeat)
        weights -= lr * currLossGrad
        count -= 1
    return weights, losses

def loss(data, labels, weights, numDataPoint):
    MSE = 0
    for index in range(numDataPoint):
        MSE += np.square(np.dot(data[index], np.transpose(weights))[0] - labels[index])
    return MSE/numDataPoint

def lossGrad(data, labels, weights, numDataPoint, numFeat):
    grad = np.zeros(numFeat)
    for index in range(numDataPoint):
        grad += (np.dot(data[index], np.transpose(weights))[0] - labels[index]) * data[index]
    return (2/numDataPoint) * grad


# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    # Your code here:

    return

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:
print(preprocess_data(load_data('IA1_train.csv'), 1, 0))

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:
#print(gd_train(preprocess_data(load_data('IA1_train.csv'), 1, 0), 1, 1))

# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



