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
    # (1) Delete the first column of house ID
    #preprocessed_data = np.delete(data, 0, 1)
    # (2) Delete the first row
    #preprocessed_data = np.delete(data, 0, 0)
    # (3) spit date
    #df = pd.DataFrame(preprocessed_data)
    #print(df)
    #return preprocessed_data
    data[["month", "day", "year"]] = data["date"].str.split("/", expand=True)
    data['year'] = data['year'].astype(int)
    data['month'] = data['month'].astype(int)
    data['day'] = data['day'].astype(int)
    preprocessed_data = pd.DataFrame(data).to_numpy()
    #print(preprocessed_data)
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
    NumFeat = np.shape(data)[1] - 1
    # Number of training data = number of rows from pre-processed data \
    NumTraning = np.shape(data)[0] 
    # Initialize weights to all zero 
    weights = np.zeros(NumFeat)
    # this array stores the loss after each iteration
    losses = []
    #we are supposed to run 4000 iterations if the loss is not convereged 
    count = 4000
    # this record loss for the current iteration
    currLoss = 1
    # termination conditions
    while count > 0 and currLoss > 1:
        # TODO: calculate current loss and gradient
        count -= 1
    return weights, losses

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


# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



