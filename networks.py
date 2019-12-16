"""**Definition of ResNet-50**"""
def resnet_model(input_shape, num_classes):
  resnet = Sequential()
  resnet.add(keras.applications.resnet.ResNet50(include_top=False, weights= None, input_tensor=None, input_shape = input_shape, pooling='max', classes=num_classes))
  resnet.add(Dense(num_classes, activation = 'softmax'))
  return resnet

"""**Definition of VGG16**"""
def vgg16_model(input_shape ,num_classes):
  vgg16 = Sequential()
  vgg16.add(keras.applications.vgg16.VGG16(include_top=False, weights= None, input_tensor=None, input_shape = input_shape, pooling='max', classes=num_classes))
  vgg16.add(Dense(num_classes, activation = 'softmax'))
  return vgg16

"""**Definition of LeNet-5**"""
def lenet_model(input_shape, num_classes):
  lenet = keras.Sequential()
  lenet.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
  lenet.add(MaxPool2D())
  lenet.add(Conv2D(filters=48, kernel_size=(5, 5), activation='relu'))
  lenet.add(MaxPool2D())
  lenet.add(Flatten())
  lenet.add(Dense(units=120, activation='relu'))
  lenet.add(Dense(units=84, activation='relu'))
  lenet.add(Dense(units=num_classes, activation = 'softmax'))
  return lenet

"""**Defines a multi-layer perceptron (MLP). Hidden units is a vector containing the number of hidden units between each layer.**"""
def MLP(hidden_units, input_shape, num_classes, dropout_rate = 0):
  MLP = Sequential()
  MLP.add(Flatten(input_shape = input_shape))
  MLP.add(Dense(hidden_units[0], activation = 'relu', input_shape = input_shape))
  MLP.add(BatchNormalization())
  MLP.add(Dropout(rate = dropout_rate))
  for i in range(len(hidden_units)-1):
    MLP.add(Dense(hidden_units[i+1], activation = 'relu'))
    MLP.add(BatchNormalization())
    MLP.add(Dropout(rate = dropout_rate))
  MLP.add(Dense(num_classes, activation = 'softmax'))
  return MLP

"""**Builds baseline ResNet-50. Returns predictions and labels**"""
def resnet(dataset):
  train_size, test_size, epochs, batch_size = get_hyperparameters('resnet')
  x_train, x_test, y_train, y_test, num_classes, H, W, D = define_dataset(dataset, train_size, test_size)
  model = resnet_model((H, W, D), num_classes)
  y_pred, accuracy = compile_fit_predict(model, x_train, y_train, x_test, y_test, epochs, batch_size)
  return y_pred, y_test

"""**Builds baseline VGG-16. Returns predictions and labels**"""
def vgg16(dataset):
  train_size, test_size, epochs, batch_size = get_hyperparameters('vgg16')
  x_train, x_test, y_train, y_test, num_classes, H, W, D = define_dataset(dataset, train_size, test_size)
  model = vgg16_model((H, W, D), num_classes)
  y_pred, accuracy = compile_fit_predict(model, x_train, y_train, x_test, y_test, epochs, batch_size)
  return y_pred, y_test

"""**Builds baseline LeNet-5. Returns predictions and labels**"""
def lenet(dataset):
  train_size, test_size, epochs, batch_size = get_hyperparameters('lenet')
  x_train, x_test, y_train, y_test, num_classes, H, W, D = define_dataset(dataset, train_size, test_size)
  model = lenet_model((H, W, D), num_classes)
  y_pred, accuracy = compile_fit_predict(model, x_train, y_train, x_test, y_test, epochs, batch_size)
  return y_pred, y_test

"""**Return hyperparameters for a given network.**"""
def get_hyperparameters(network):
  if network is 'lenet':
    train_size = 900
    test_size = 40000
    epochs = 10
    batch_size = 64
  elif network is 'vgg16':
    train_size = 5000
    test_size = 40000
    epochs = 10
    batch_size = 64
  elif network is 'resnet':
    train_size = 6000
    test_size = 40000
    epochs = 10
    batch_size = 64
  elif network is 'MLP':
    train_size = 100 #****Should be changed*****
    test_size = 100
    batch_size = 64
    epochs = 10
  else:
    raise Exception('Unimplemented network')
  return train_size, test_size, epochs, batch_size


