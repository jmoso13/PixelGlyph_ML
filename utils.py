from keras.layers import Lambda, Input, Dense, Dropout, LeakyReLU, BatchNormalization, Layer, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, GlobalAveragePooling2D
from keras.activations import linear, sigmoid
from keras.models import Model, load_model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np
from PIL import Image


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_conv_vae_dict(original_dim, l, model_name, e_f=[32,64,64,64], e_ks=[3,3,3,3], e_s=[1,2,2,2], d_f=[64,64,32,3], d_ks=[3,3,3,3], d_s=[2,2,2,1], loss_weights=[0.5, 0.5], dropout_rate=0.0, batch_norm=False, verbose=True):
  # Input Layer
  input_shape = original_dim
  inputs = Input(shape=input_shape, name='encoder_input')
  last_layer = inputs
  if dropout_rate:
    inputs_dropout = Dropout(rate=dropout_rate)(inputs)
    last_layer = inputs_dropout
  # Hidden Layers
  h_l_dict = dict()
  h_l = len(e_f)
  for x in range(h_l):
    layer_name = 'encoder_layer_%s'%(x+1)
    h_l_dict[layer_name] = Conv2D(filters=e_f[x], kernel_size=e_ks[x], strides=e_s[x], padding='same', name=layer_name)(last_layer)
    h_l_dict[layer_name] = LeakyReLU(name=layer_name+'_leaky_relu')(h_l_dict[layer_name])
    if batch_norm:
      h_l_dict[layer_name] = BatchNormalization(name=layer_name+'_batch_norm')(h_l_dict[layer_name])
    if dropout_rate:
      h_l_dict[layer_name] = Dropout(rate=dropout_rate, name=layer_name+'_dropout')(h_l_dict[layer_name])
    last_layer = h_l_dict[layer_name]
  # VAE Layer
  shape_before_flattening = K.int_shape(last_layer)[1:]
  last_layer = Flatten()(last_layer)
  z_mean = Dense(l, name='z_mean')(last_layer)
  z_log_var = Dense(l, name='z_log_var')(last_layer)
  z = Lambda(sampling, output_shape=(l,), name='z')([z_mean, z_log_var])
  # Instantiate encoder model
  encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
  # DECODER
  last_layer = Dense(np.prod(shape_before_flattening), name='upscale_latent')(z)
  last_layer = Reshape(shape_before_flattening)(last_layer)
  for x in range(len(d_f) - 1):
    layer_name = 'decoder_layer_%s'%(x+1)
    h_l_dict[layer_name] = Conv2DTranspose(filters=d_f[x], kernel_size=d_ks[x], strides=d_s[x], padding='same', name=layer_name)(last_layer)
    h_l_dict[layer_name] = LeakyReLU(name=layer_name+'_leaky_relu')(h_l_dict[layer_name])
    if batch_norm:
      h_l_dict[layer_name] = BatchNormalization(name=layer_name+'_batch_norm')(h_l_dict[layer_name])
    if dropout_rate:
      h_l_dict[layer_name] = Dropout(rate=dropout_rate, name=layer_name+'_dropout')(h_l_dict[layer_name])
    last_layer = h_l_dict[layer_name]        
  outputs = Conv2DTranspose(filters=d_f[-1], kernel_size=d_ks[-1], strides=d_s[-1], padding='same', activation='tanh', name='outputs')(last_layer)
  # Model to train
  vae = Model(inputs, outputs, name=model_name)
  # Defining Loss
  reconstruction_loss = K.mean(K.square(inputs-outputs), axis=[1,2,3])
  reconstruction_loss *= np.prod(original_dim)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(loss_weights[0]*reconstruction_loss + loss_weights[1]*kl_loss)
  get_losses = K.function([inputs], [vae_loss, reconstruction_loss, kl_loss])
  get_rls = K.function([inputs], [reconstruction_loss])
  vae.add_loss(vae_loss)
  vae.compile(optimizer='adam')
  if verbose:
    print(encoder.summary())
    print(vae.summary())

  return {'vae': vae, 
          'encoder': encoder,
          'get_losses_function': get_losses,
          'get_rls_function': get_rls
          }


def build_decoder(vae, decoder_input):
  layer_list = list()
  for l in reversed(vae.layers):
    if l.name == 'z':
      break
    layer_list.append(l)
  decoder_output = decoder_input
  for l in reversed(layer_list):
    print(l.name)
    decoder_output = l(decoder_output)
  de = Model(decoder_input, decoder_output)
  return de


def project(decoder, latent_point):
  return np.squeeze(decoder.predict(np.array([latent_point])) * 255., axis=0).astype('uint8')


def project_matrix(decoder, latent_point, matrix_means=None, rounding=True):
  if rounding:
    if matrix_means is not None:
      return np.multiply(np.greater(np.squeeze(decoder.predict(np.array([latent_point])) + 0.5), matrix_means).astype(int), (matrix_means > 0).astype(int)).astype('uint8')
    else:
      return np.squeeze((decoder.predict(np.array([latent_point])) + 0.5).round(), axis=0).astype('uint8')
  else:
    return np.squeeze(np.clip(decoder.predict(np.array([latent_point])) + 0.5, a_min = 0, a_max=1) * 255., axis=0).astype('uint8')


def save_random_pred(vae, x_test, model_name):
  r = np.random.choice(range(x_test.shape[0]))
  r = x_test[r]
  s = r.shape
  if s[-1] == 1:
    if np.prod(s) == 50:
      im_og = Image.fromarray(((r + 0.5) * 255.).astype('uint8').reshape(s[:-1]), 'L')
      im_pred = Image.fromarray((np.clip(vae.predict(r.reshape((1,)+s)) + 0.5, a_min=0, a_max=1) * 255.).reshape(s).astype('uint8').reshape(s[:-1]), 'L')
    else:
      im_og = Image.fromarray((r * 255.).astype('uint8').reshape(s[:-1]), 'L')
      im_pred = Image.fromarray((vae.predict(r.reshape((1,)+s)) * 255.).reshape(s).astype('uint8').reshape(s[:-1]), 'L')
  else:
    im_og = Image.fromarray((r * 255.).astype('uint8'))
    im_pred = Image.fromarray((vae.predict(r.reshape((1,)+s)) * 255.).reshape(s).astype('uint8'))
  im_og.save(f'{model_name}_og.png')
  im_pred.save(f'{model_name}_pred.png')


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
  return EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience), ModelCheckpoint(f'models/best_{model_name}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)