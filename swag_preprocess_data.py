"""
Performs preprocessing of a dataset before training.

1. Split the dataset into train/valid/test.
2. Preprocess the input samples to the desired input format.
3. Save processed data to train, valid, and test directories.
"""

import argparse
import numpy as np
import os
import tensorflow as tf
import keras
from keras import applications
from keras.datasets import mnist
from keras import utils as keras_utils
import utils


def parse_arguments():
    """
    Parses input arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", dest="save_path", metavar="SAVE PATH", default=None,
                        help="Path to save the processed data in.")
    parser.add_argument("--data_set", dest="data_set", metavar="DATA SET", default=None,
                        help="Specify what dataset you want to preprocess (MNIST, MED, cifar10)")
    parser.add_argument("--train_size", dest="train_size", metavar="TRAIN SIZE", default=None,
                        help="Specify train size.", type = int)
    parser.add_argument("--test_size", dest="test_size", metavar="TEST SIZE", default=None,
                        help="Specify test size", type = int)  
 
    args = parser.parse_args()
    train_frac = 0.9
    #valid_frac = 0.1
    assert args.save_path is not None, "Save path must be specified."
    assert args.data_set in ["MNIST", "MED"], "Invalid choice of dataset. Must be MNIST or MED"

    if os.path.exists(args.save_path):
        response = input("Save path already exists. Previous data may be overwritten. Continue? (y/n) >> ")
        if response in ["n", "N", "no"]:
            exit()
    else:
        response = input("Save path does not exist. Create it? (Y/n) >> ")
        if response in ["n", "N", "no"]:
            exit()
        os.makedirs(os.path.join(args.save_path, ""))

    return args

def normalize_images(images, H, W):
    images = np.reshape(images, (-1, H * W))/255.0
    return np.reshape(images, (-1, H, W))

def load_mnist(train_size, test_size):
  if (train_size + test_size)>70000: #Raise an exception if too much data is requested
    raise Exception('Not enough data')

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  #Create array with all data for x & y
  x = np.concatenate((x_train, x_test),axis=0)
  y = np.concatenate((y_train, y_test),axis=0)

  W = x.shape[1] #Width of image in pixels
  H = x.shape[2] #Height 
  D = 1 #Third dimension of the data, e.g. 1 if black-white image
  x = normalize_images(x, H, W)

  #Reshape
  x = x.reshape(-1, H, W, 1).astype('float32')
  y = keras_utils.to_categorical(y) # encode one-hot vector

  #Zero-padding
  x = np.pad(x, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
  y = y.astype('float32')


  #Divide into train and test dataset
  x_train = x[:train_size]
  y_train = y[:train_size]
  x_test = x[train_size:train_size+test_size]
  y_test = y[train_size:train_size+test_size]

  num_classes = y_train.shape[1]

  W = x.shape[1] #Width of image in pixels
  H = x.shape[2] #Height 

  return (x_train, y_train), (x_test, y_test), num_classes, H, W, D


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    save_path = os.path.join(args.save_path, "")

    # Read the dataset
    if args.data_set == "MNIST":
        (x_train, y_train), (x_test, y_test), num_classes, H, W, D = load_mnist(args.train_size, args.test_size)


    # Save data to specified save path
    np.save(save_path + "x_train.npy", x_train)
    np.save(save_path + "y_train.npy", y_train)
    np.save(save_path + "x_test.npy", x_test)
    np.save(save_path + "y_test.npy", y_test)

    print("Processed data was saved to {} in .npy files.".format(save_path))