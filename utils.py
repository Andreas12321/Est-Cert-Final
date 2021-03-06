'''Import packages'''
import measures

import matplotlib.pyplot as plt
import os, os.path
import time
import keras
import numpy as np
from keras import optimizers
from keras.datasets import mnist
from keras import utils as keras_utils


"""**Normalize images**"""
def normalize_images(images, H, W):
    images = np.reshape(images, (-1, H * W))/255.0
    return np.reshape(images, (-1, H, W))

"""**Load MNIST with no validation data**"""
def load_mnist_no_val(train_size, test_size, padding):
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
    if padding:
        x = np.pad(x, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        
    y = y.astype('float32')

    #Divide into train, validation and test dataset
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[train_size:train_size+test_size]
    y_test = y[train_size:train_size+test_size]

    num_classes = y_train.shape[1]

    W = x.shape[1] #Width of image in pixels
    H = x.shape[2] #Height 

    return (x_train, y_train), (x_test, y_test), num_classes, H, W, D

"""**Plot first image from data**"""
def plot_image(data):
    first_image = data[0][0][0]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

"""**Compiles, trains, evaluates and predict for a given model. Returns predictions and labels.**"""
def compile_fit_predict(model, x_train, y_train, x_test, y_test, epochs, batch_size, verbose = 1):
    test_batch_size = 32
    optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay = 1e-4) #kolla upp detta!
    model.compile(optimizer = optimizer,loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #model.summary()
    start = time.time()
    model.fit(x = x_train, y = y_train, epochs = epochs, batch_size = batch_size, verbose = verbose)
    end = time.time()
    print("Training time: ", end - start, "seconds")
    start = time.time()
    loss_metrics = model.evaluate(x_test,y_test, batch_size = test_batch_size)
    end = time.time()
    print("Evaluation time: ", end - start, "seconds")
    print("Loss metrics:", loss_metrics)
    y_pred = model.predict(x_test)
    return y_pred, loss_metrics[1]

"""**Loads the chosen dataset**"""
def define_dataset(dataset, train_size, test_size):
    if(dataset == 'mnist'):
        (x_train, y_train), (x_test, y_test), num_classes, H, W, D = load_mnist_no_val(train_size, test_size, True)
    elif(dataset == 'oral'):
        raise Exception('Cannot choose sizes yet')
        #x_train, x_test, y_train, y_test, num_classes, H, W, D = load_real_data()
    else: 
        raise Exception('Dataset not found')
    return x_train, x_test, y_train, y_test, num_classes, H, W, D

"""**Returns the logits** https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/uq_utils.py"""
def np_inverse_softmax(x):
    """Invert a softmax operation over the last axis of a np.ndarray."""
    return np.log(x / x[..., :1])

"""**Save data to file. Assumes one folder per network. One file in each network for every run of the method.**"""
def save_data(method, network, y_pred, y_test):
    NLL, brier, ECE, AECE, AMCE, ece_confidence, ece_accuracy, adaptive_confidence, adaptive_accuracy = measures.certainty_measures(y_pred, y_test)
    a = {'method': method, 'network': network, 'y_pred': y_pred, 'y_test' : y_test, 'NLL': NLL, 'brier': brier, 
        'ECE': ECE, 'AECE': AECE, 'AMCE': AMCE, 'ece_confidence': ece_confidence, 'ece_accuracy': ece_accuracy, 
        'adaptive_confidence': adaptive_confidence, 'adaptive_accuracy': adaptive_accuracy}
    if not os.path.exists(network):  #Create directory if needed
        os.mkdir(network)
    np.savez_compressed(network + "/" + method +  "_result.npz", **a)

"""**Loads results from files for a network. Saves in dictionary, method is key.**"""
def load_results(network):
    result = {}
    methods = ['baseline', 'temp_scale', 'SWAG', 'ensemble']
    for method in methods:
        path = network + "/" + method + '_result.npz'
        if(os.path.exists(path)):
            x = np.load(path, allow_pickle=True)  
            result[method] = x      
    return result

"""**Reads predictions and correct labels. Returns results in same format as other methods.**"""
def load_swag_results(network):
    path = "SWAG_" + network + "_result_plot.npz"
    if not os.path.exists(path):
        raise Exception("Path does not exist!")
    x = np.load(path, allow_pickle=True)  
    y_pred = x['y_pred']
    y_test = x['y_test']

    save_data('SWAG', network, y_pred, y_test)

"""**Defines piece-wise constant learning rate schedule for SWAG. **"""
def schedule(num_epochs, num_train_sample, batch_size, swa_start,  swa_lr, lr_init = 0.01, swa = True): #Default value of lr_init & swa_lr from original implementation
    num_batches = num_train_sample // batch_size
    learning_rate_vector = []
    calls = num_epochs * num_batches
    if calls == 0:
        raise Exception('Not enough data.')
    for call in range(1,calls+1):
        epoch = ceildiv(call, num_batches)
        learning_rate_vector.append(calc_learn_rate(epoch, swa = swa, swa_start= swa_start, swa_lr = swa_lr, lr_init = lr_init))
    
    #Get the unique values and indices from the lr_vector
    unique_vector, indices = np.unique(learning_rate_vector, return_index=True)
    indices = indices[1:] #Remove the 0
    
    return unique_vector.tolist(), indices.tolist()

"""**Calsulates learning rate.**"""
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