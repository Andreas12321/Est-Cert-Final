########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np

DROPOUT_RATE = 0.0
D = 1


class lenet:

    def __init__(self, imgs, n_classes, weights=None, sess=None, dropout=DROPOUT_RATE):
        """
        Initializes the VGG16 network.
        """

        self.imgs = imgs
        self.n_classes = n_classes
        self.conv_initializer = tf.contrib.layers.variance_scaling_initializer()  # The initilization
        self.fc_initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.weight_keys = []
        self.dropout = dropout
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights_from_file(weights, sess)

        conv_out_shape = int(np.prod(self.pool2.get_shape()[1:]))

        self.lenet_VAR_DIMS = {         
            "conv1_1_W": (5, 5, D, 32), "conv1_1_b": (32),
            "conv1_2_W": (5, 5, 32, 32), "conv1_2_b": (32),
            "conv2_1_W": (5, 5, 32, 48), "conv2_1_b": (48),
            "conv2_2_W": (5, 5, 48, 48), "conv2_2_b": (48),
            "fc1_W": (conv_out_shape, 120), "fc1_b": (120),
            "fc2_W": (120, 84), "fc2_b": (84),
            "fc3_W": (84, self.n_classes), "fc3_b": (self.n_classes)
        }

    def convlayers(self):
        """
        Adds the convolution layers to the network.
        """
        self.parameters = []

        images = self.imgs
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            images = tf.nn.dropout(images, rate=self.dropout)
            kernel = tf.Variable(self.conv_initializer([5, 5, D, 32]), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv1_1_W", "conv1_1_b"]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            self.conv1_1 = tf.nn.dropout(self.conv1_1, rate=self.dropout)
            kernel = tf.Variable(self.conv_initializer([5, 5, 32, 32]), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv1_2_W", "conv1_2_b"]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            self.conv1_2 = tf.nn.dropout(self.conv1_2, rate=self.dropout)
            kernel = tf.Variable(self.conv_initializer([5, 5, 32, 48]), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv2_1_W", "conv2_1_b"]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            self.conv2_1 = tf.nn.dropout(self.conv2_1, rate=self.dropout)
            kernel = tf.Variable(self.conv_initializer([5, 5, 48, 48]), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            self.weight_keys += ["conv2_2_W", "conv2_2_b"]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

    def fc_layers(self):
        """
        Adds the fully connected layers to the network.
        """
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool2.get_shape()[1:]))
            fc1w = tf.Variable(self.fc_initializer([shape, 120]), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[120], dtype=tf.float32),
                               trainable=True, name='biases')
            pool2_flat = tf.reshape(self.pool2, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1l = tf.nn.dropout(fc1l, rate=self.dropout)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
            self.weight_keys += ["fc1_W", "fc1_b"]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(self.fc_initializer([120, 84]), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[84], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            fc2l = tf.nn.dropout(fc2l, rate=self.dropout)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]
            self.weight_keys += ["fc2_W", "fc2_b"]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(self.fc_initializer([84, self.n_classes]), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.n_classes], dtype=tf.float32),
                               trainable=True, name='biases')
            fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.fc3l = tf.nn.dropout(fc3l, rate=self.dropout)
            self.parameters += [fc3w, fc3b]
            self.weight_keys += ["fc3_W", "fc3_b"]

    def load_weights_from_file(self, weight_file, sess):
        """
        Initializes the weights of the network to the values in weight_file.
        """
        weight_dict = np.load(weight_file)
        self.load_weights(weight_dict, sess)

    def load_weights(self, weight_dict, sess):
        """
        Initializes the weights of the network to the values in weight_dict.
        (weight_dict needs to be a dictionary structured as what is returned by
        self.get_weights).
        """
        keys = sorted(weight_dict.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weight_dict[k]))

    def save_weights(self, weight_path, weight_file_name, sess):
        """
        Saves the current weights of the network to a file with the name
        weight_file_name in weight_path.
        """
        weight_dict = self.get_weights(sess)
        np.savez(weight_path + weight_file_name, **weight_dict)

    def get_weights(self, sess):
        """
        Returns the current values of all weights in the network
        in a dictionary with the same keys as self.weight_keys.
        """
        keys = sorted(self.weight_keys)
        weight_dict = {}
        for i, k in enumerate(keys):
            weight_dict[k] = sess.run(self.parameters[i])

        return weight_dict

    def get_weights_flat(self, sess):
        """
        Returns the current values of all weights in the network
        in a single flattened vector.

        Weights are ordered alphabetically after their key (e.g. conv1_1_W) and
        flattened in row major order.
        """
        weight_dict = self.get_weights(sess)
        keys = sorted(weight_dict.keys())
        weight_vector = []

        for key in keys:
            weight_vector.append(weight_dict[key].flatten())

        return np.concatenate(weight_vector)

    def unflatten_weights(self, weight_vector):
        """
        Takes a vector of weights (structured as the return value of self.get_weights_flat)
        and "unflattens" them into a weight_dict (like the one returned by get_weights).
        """
        keys = sorted(self.lenet_VAR_DIMS.keys())
        weight_dict = {}
        slice_index = 0

        for key in keys:
            dims = self.lenet_VAR_DIMS[key]
            size = np.prod(dims)
            values = weight_vector[slice_index: slice_index + size]
            slice_index += size

            weight_dict[key] = values.reshape(dims)

        return weight_dict

