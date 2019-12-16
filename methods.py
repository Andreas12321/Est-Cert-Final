'''Import packages'''
import utils
import networks
from networks import lenet_model
from networks import vgg16_model
from networks import resnet_model

import keras
from keras import optimizers

import numpy as np

"""**Builds ensemble using MLP. Returns predictions and labels.**"""
def ensemble(network, dataset):
  ensemble_members = 10
  count = 0
  it = 0
  train_size, test_size, epochs, batch_size = networks.get_hyperparameters(network)
  x_train, x_test, y_train, y_test, num_classes, H, W, D = utils.define_dataset(dataset, train_size, test_size)
  optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay = 1e-4) #kolla upp detta!
  prob_temp = np.zeros((test_size, num_classes))
  while count < ensemble_members:
    if network == 'MLP':
      input_shape = x_train.shape[1:]
      hidden_units = [200, 200, 200]
      model = networks.MLP(hidden_units, input_shape, num_classes, dropout_rate = 0)
    elif network in ['lenet', 'vgg16', 'resnet']:
      model = globals()[network + "_model"]((H, W, D), num_classes)
    y_pred, accuracy = utils.compile_fit_predict(model, x_train, y_train, x_test, y_test, epochs, batch_size, verbose = 0)
    if accuracy < 0.2:
      print("Ensemble member removed. Current number of ensemble members: ", count, '\n')
      continue 
    prob_temp += y_pred
    count += 1
    print("Ensemble member kept. Current number of ensemble members: ", count, '\n')
  y_pred = prob_temp/float(ensemble_members)
  return y_pred, y_test

"""**Temperature scaling using session** https://github.com/ondrejba/tf_calibrate/blob/master/calibration.py"""
def temperature_scale(logits, session, valid_labels, learning_rate=0.01, num_steps=50):
  """
  Calibrate the confidence prediction using temperature scaling.
  :param logits:          Outputs of the neural network before softmax.
  :param session:         Tensorflow session.
  :param x_pl:            Placeholder for the inputs to the NN.
  :param y_pl:            Placeholder for the label for the loss.
  :param valid_data:      Validation inputs.
  :param valid_labels:    Validation labels.
  :return:                Scaled predictions op.
  """

  temperature = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32, name="temperature")
  scaled_logits = logits / temperature

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=scaled_logits))

  opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
  opt_step = opt.minimize(loss, var_list=[temperature])

  session.run(tf.variables_initializer([temperature]))
  session.run(tf.variables_initializer([opt.get_slot(var, name) for name in opt.get_slot_names() for var in [temperature]]))

  for i in range(num_steps):
    session.run(opt_step)

  return temperature

"""**Applies temperature scaling. Returns the scaled predictions and temperature.**"""
def run_temp_scale(y_pred,y_test):
    logits_test = utils.np_inverse_softmax(y_pred)

    #Initialize sesstion and run
    init_op = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init_op)
        sparse_labels = np.argmax(y_test, axis=1) 
        temp = temperature_scale(logits_test, sess, sparse_labels, learning_rate=0.01, num_steps=50)
        print("Temperature: ", sess.run(temp))
        scaled_logits = logits_test / temp

        #Evaluates the values of temp and scaled_predictions
        temp = sess.run(temp)
        scaled_predictions = sess.run(tf.nn.softmax(scaled_logits))
        
    return scaled_predictions, temp