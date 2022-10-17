# CS 534
# AI1 skeleton code
# By Quintin Pope

import numpy as np
import pandas as pd
import matplotlib

current_year = 2022

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
    data = data.drop(columns=['id', 'date'])
    print(data)
    return data

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

    return weights, losses

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    # Your code here:

    return

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:
print(preprocess_data(load_data('IA1_train.csv'), 1, 1))

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:


# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



