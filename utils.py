from keras.layers import Lambda, Input, Dense, Dropout, LeakyReLU, BatchNormalization, Layer, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, GlobalAveragePooling2D
from keras.activations import linear, sigmoid
from keras.models import Model, load_model
from keras.datasets import mnist, fashion_mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


def get_conv_multilabel_nn(original_dim, num_classes, model_name, num_filters=[32,64,64,64], kernel_size=[3,3,3,3], stride=[1,2,2,2], dropout_rate=0.0, batch_norm=False, max_pooling=None):
  # Input Layer
  input_shape = original_dim
  inputs = Input(shape=input_shape, name='encoder_input')
  last_layer = inputs
  if dropout_rate:
    inputs_dropout = Dropout(rate=dropout_rate)(inputs)
    last_layer = inputs_dropout
  # Hidden Layers
  h_l_dict = dict()
  h_l = len(num_filters)
  for x in range(h_l):
    layer_name = 'encoder_layer_%s'%(x+1)
    h_l_dict[layer_name] = Conv2D(filters=num_filters[x], kernel_size=kernel_size[x], strides=stride[x], padding='same', name=layer_name)(last_layer)
    if max_pooling:
      h_l_dict[layer_name] = MaxPooling2D(pool_size=max_pooling[x])(h_l_dict[layer_name])
    if batch_norm:
      h_l_dict[layer_name] = BatchNormalization(name=layer_name+'_batch_norm')(h_l_dict[layer_name])
    h_l_dict[layer_name] = LeakyReLU(name=layer_name+'_leaky_relu')(h_l_dict[layer_name])
    if dropout_rate:
      h_l_dict[layer_name] = Dropout(rate=dropout_rate, name=layer_name+'_dropout')(h_l_dict[layer_name])
    last_layer = h_l_dict[layer_name]
  if max_pooling:
    last_layer = GlobalAveragePooling2D()(last_layer)
  else:
    last_layer = Flatten()(last_layer)
  outputs = Dense(num_classes, activation='sigmoid', name='outputs')(last_layer)
  nn = Model(inputs, outputs, name=model_name)
  nn.compile(loss='binary_crossentropy',
              optimizer='adam')
  print(nn.summary())
  return nn


def get_model_callbacks(model_name, patience=100):
  return EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience), ModelCheckpoint(f'best_{model_name}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)