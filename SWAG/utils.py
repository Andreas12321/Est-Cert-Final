"""
Utility functions.
"""

import numpy as np
import pickle


def unpickle(file_name):
    """
    Unpickles a pickled Python-object.
    """

    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def shuffle_data(X, y, one_hot_labels=True):
    """
    Shuffles the data in X, y with the same random permutation.
    Use one_hot_labels to specify whether the labels in y are
    in one-hot or index format.
    """

    n_examples = X.shape[0]

    perm = np.random.permutation(n_examples)
    X = X[perm, :, :]
    if one_hot_labels:
        y = y[perm, :]
    else:
        y = y[perm]

    return X, y


def split_data(X, y, train_frac, valid_frac, test_frac, shuffle=True):
    """
    Splits data into train/valid/test-sets according to the specified fractions.
    If shuffle is True, data is shuffled before splitting.
    """

    np.random.seed(1)

    assert train_frac + valid_frac + test_frac == 1, "Train/valid/test data fractions do not sum to one"

    n_examples = X.shape[0]

    # Shuffle
    if shuffle:
        X, y = shuffle_data(X, y)

    # Split
    ind_1 = int(np.round(train_frac * n_examples))
    ind_2 = int(np.round(ind_1 + valid_frac * n_examples))

    X_train = X[0:ind_1, :, :]
    y_train = y[0:ind_1, :]
    X_valid = X[ind_1:ind_2, :, :]
    y_valid = y[ind_1:ind_2, :]
    X_test = X[ind_2:, :, :]
    y_test = y[ind_2:, :]

    assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == n_examples, "Data split failed"

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def index_to_one_hot(y, n_classes):
    """
    Converts a list of indices to one-hot encoding.
    Example: y = [1, 0, 3] => np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    """
    one_hot = np.eye(n_classes)[y]

    return one_hot


def schedule(num_epochs, num_train_sample, batch_size, swa_start,  swa_lr, lr_init = 0.01, swa = True): #Default value of lr_init & swa_lr from original implementation
    num_batches = num_train_sample // batch_size
    learning_rate_vector = []

    calls = num_epochs * num_batches
    for call in range(1,calls+1):
        epoch = ceildiv(call, num_batches)
        learning_rate_vector.append(calc_learn_rate(epoch, swa = swa, swa_start= swa_start, swa_lr = swa_lr, lr_init = lr_init))
    
    #Get the unique values and indices from the lr_vector
    unique_vector, indices = np.unique(learning_rate_vector, return_index=True)
    indices = indices[1:] #Remove the 0
    
    return unique_vector.tolist(), indices.tolist()

def calc_learn_rate(epoch, swa, swa_start, swa_lr, lr_init):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else lr_init
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def ceildiv(a,b):
    return -(-a//b)

def save_data(method, network, y_pred, y_test):
    np.savez_compressed(method + "_" + network +  "_result_plot.npz", y_pred = y_pred, y_test = y_test)