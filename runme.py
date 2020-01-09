'''Import packages'''
import methods
import networks
from networks import lenet
from networks import vgg16
from networks import resnet
import utils

import tensorflow as tf
import keras
import numpy as np
np.random.seed(310) # NumPy
import random
random.seed(420) # Python
from tensorflow import set_random_seed
set_random_seed(440) # Tensorflow

from keras import applications
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import asarray

from keras import optimizers
from keras import utils as keras_utils
from keras.layers import (Conv2D, BatchNormalization, MaxPool2D, Dense, Dropout, Flatten)

from keras.models import (Model, Sequential)
import sklearn.metrics as sk_met

import os, os.path
import time
import argparse

from AdaptiveBinning import AdaptiveBinning


def parse_arguments():
    """
    Parses input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", dest="network", metavar="NETWORK", default=None,
                        help="Choose network.")
    parser.add_argument("--dataset", dest="dataset", metavar="DATASET", default=None,
                        help="Dataset.")
    parser.add_argument("--method", dest="method", metavar="METHOD", default=None,
                        help="Method.")                    
    
    args = parser.parse_args()
    return args

"""**Definition that creates and compiles the selected network. Specify train size etc.**"""
def run_model(network, dataset, method):  
  if dataset not in ['mnist']:
    raise Exception('Dataset not available')
  if method in ['baseline', 'temp_scale']:
    if network not in ['lenet', 'vgg16', 'resnet']:
      raise Exception('Unimplemented network')
    y_pred, y_test = globals()[network](dataset)
    if method is 'temp_scale':
      y_pred, temp = methods.run_temp_scale(y_pred, y_test)
  elif method is 'ensemble':
    if network not in ['lenet', 'vgg16', 'resnet', 'MLP']:
      raise Exception('Unimplemented network')
    y_pred, y_test = methods.ensemble(network, dataset)
  else:
    raise Exception('Unimplemented method')
  return y_pred, y_test

"""**Main, trains network and returns predictions**"""
def main(network, dataset, method):
  y_pred, y_test = run_model(network, dataset, method)
  utils.save_data(method, network, y_pred, y_test)
  return y_pred, y_test
#network = 'lenet'
#dataset = 'mnist'
#method = 'baseline'
args = parse_arguments()
print('In progress: ', args.method)

y_pred, y_test = main(args.network, args.dataset, args.method)
