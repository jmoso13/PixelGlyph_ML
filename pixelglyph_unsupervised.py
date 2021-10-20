from keras.layers import Lambda, Input, Dense, Dropout, LeakyReLU, BatchNormalization, Layer, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, GlobalAveragePooling2D
from keras.activations import linear, sigmoid
from keras.models import Model, load_model
from keras.datasets import mnist, fashion_mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from scipy.stats import multivariate_normal, norm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf
from scipy.stats import gaussian_kde


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
import time
import pickle
import datetime
from sklearn.metrics import roc_auc_score
import pickle as pkl
# import xgboost as xgb
# import shap
from sklearn.manifold import TSNE
from PIL import Image

glyph_list = list()
for g in glob.glob('small_glyphs/*.png'):
  glyph_list.append(np.asarray(Image.open(g)))

x_train_msk = np.random.choice(np.arange(len(glyph_list)), int(0.8*len(glyph_list)), replace=False)
x_test_msk = [i for i in range(len(glyph_list)) if i not in x_train_msk]
assert len(x_train_msk) + len(x_test_msk) == 10000
x_train = np.array(glyph_list)[x_train_msk].astype(float)/255.
x_test = np.array(glyph_list)[x_test_msk].astype(float)/255.


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


def dim_decay_func(x, u, l, n):
  return int(np.rint(-np.power(x, n) * (u - l) + u))


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
    if batch_norm:
      h_l_dict[layer_name] = BatchNormalization(name=layer_name+'_batch_norm')(h_l_dict[layer_name])
    h_l_dict[layer_name] = LeakyReLU(name=layer_name+'_leaky_relu')(h_l_dict[layer_name])
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
  for x in range(h_l-1):
    layer_name = 'decoder_layer_%s'%(x+1)
    h_l_dict[layer_name] = Conv2DTranspose(filters=d_f[x], kernel_size=d_ks[x], strides=d_s[x], padding='same', name=layer_name)(last_layer)
    if batch_norm:
      h_l_dict[layer_name] = BatchNormalization(name=layer_name+'_batch_norm')(h_l_dict[layer_name])
    h_l_dict[layer_name] = LeakyReLU(name=layer_name+'_leaky_relu')(h_l_dict[layer_name])
    if dropout_rate:
      h_l_dict[layer_name] = Dropout(rate=dropout_rate, name=layer_name+'_dropout')(h_l_dict[layer_name])
    last_layer = h_l_dict[layer_name]        
  outputs = Conv2DTranspose(filters=d_f[-1], kernel_size=d_ks[-1], strides=d_s[-1], padding='same', activation='sigmoid', name='outputs')(last_layer)
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


#VAE
lowest_dim = 2
model_name = 'pixelglyph_ae'
g_dim = 40

vae_dict = get_conv_vae_dict((g_dim,g_dim,3), lowest_dim, model_name, loss_weights=[0.6, 0.4], batch_norm=True, dropout_rate=0.0625)
vae = vae_dict['vae']
encoder = vae_dict['encoder']
get_losses = vae_dict['get_losses_function']


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint('best_%s.h5'%model_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
# Hyperparameters
batch_size = 64
epochs = 5000

vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, None),
        callbacks=[es, mc])

vae.load_weights(f'best_{model_name}.h5')


# Sample
decoder_input = Input(shape=(2,)) 
def build_decoder(vae):
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
  return (decoder.predict(np.array([latent_point])) * 255.).reshape((g_dim,g_dim,3)).astype('uint8')


def save_og_pred(vae, og):
  im_og = Image.fromarray((og * 255.).astype('uint8'))
  im_pred = Image.fromarray((vae.predict(og.reshape(-1,g_dim,g_dim,3)) * 255.).reshape((g_dim,g_dim,3)).astype('uint8'))
  im_og.save(f'{model_name}_og.png')
  im_pred.save(f'{model_name}_pred.png')


save_og_pred(vae,x_train[0])

decoder = build_decoder(vae)
points = np.linspace(-3, 3, 50)
full_pic = np.zeros((g_dim*len(points), g_dim*len(points), 3))
for i, y in enumerate(points):
  for j, x in enumerate(points):
    print(x,y)
    num = project(decoder, [x,y])
    full_pic[g_dim*len(points)-(i*g_dim+g_dim):g_dim*len(points)-i*g_dim, j*g_dim:j*g_dim+g_dim] = num

im = Image.fromarray(full_pic.astype('uint8'))
im.save(f'{model_name}_latent_space.png')



# MATRIX ONLY
with open('data/pixeldict.pkl','rb') as f:
  pixeldict = pkl.load(f)

matrices = np.array([pixeldict[i+1][0] for i in range(len(pixeldict))])
x_train_ids = np.random.choice(matrices.shape[0], int(0.8*matrices.shape[0]), replace=False)
x_test_ids = np.array([i for i in range(matrices.shape[0]) if i not in x_train_ids])
x_train = matrices[x_train_ids]
x_test = matrices[x_test_ids]
x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

lowest_dim = 2
model_name = 'pixelglyph_ae'
mat_shape = matrices[0].shape

vae_dict = get_conv_vae_dict((*mat_shape,1), lowest_dim, model_name, e_f=[32,32,8], e_ks=[2,2,3], e_s=[1,1,1], d_f=[8,32,1], d_ks=[3,2,2], d_s=[1,1,1], loss_weights=[0.96, 0.04], batch_norm=True)
vae = vae_dict['vae']
encoder = vae_dict['encoder']
get_losses = vae_dict['get_losses_function']


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint('best_%s.h5'%model_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
# Hyperparameters
batch_size = 32
epochs = 5000

vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, None),
        callbacks=[es, mc])

vae.load_weights(f'best_{model_name}.h5')

print(x_train[1].flatten() - vae.predict(x_train[[1]]).flatten().round(2))


decoder_input = Input(shape=(2,)) 
def build_decoder(vae):
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

def project(decoder, latent_point, matrix_means=None, rounding=True):
  if rounding:
    if matrix_means is not None:
      return (np.greater(decoder.predict(np.array([latent_point])).round(3).reshape((10,5)), matrix_means).astype(int) * 255.).astype('uint8')
    else:
      return (decoder.predict(np.array([latent_point])).round() * 255.).reshape((10,5)).astype('uint8')
  else:
    return (decoder.predict(np.array([latent_point])) * 255.).reshape((10,5)).astype('uint8')

matrix_means = matrices.mean(0)

decoder = build_decoder(vae)
points = np.linspace(-3, 3, 100)
full_pic = np.zeros((10*len(points), 10*len(points)))
for i, y in enumerate(points):
  for j, x in enumerate(points):
    print(x,y)
    num = project(decoder, [x,y], rounding=False)
    num = np.append(num,np.flip(num,1),1)
    full_pic[10*len(points)-(i*10+10):10*len(points)-i*10, j*10:j*10+10] = num

im = Image.fromarray(full_pic.astype('uint8'))
im.save(f'{model_name}_mat_latent_space_raw.png')


def save_og_pred(vae, og):
  im_og = Image.fromarray((og.reshape(10,5) * 255.).astype('uint8'))
  im_pred = Image.fromarray((vae.predict(og.reshape(1,10,5,1)).round() * 255.).reshape((10,5)).astype('uint8'))
  im_og.save(f'{model_name}_og.png')
  im_pred.save(f'{model_name}_pred.png')

save_og_pred(vae,matrices[9429])

rarest = 0
longest = 0
for i, m in enumerate(matrices):
  print(i)
  pos = encoder.predict(m.reshape(1,10,5,1))[0]
  line = np.sqrt(np.square(pos[0][0]) + np.square(pos[0][1]))
  if line > longest:
    rarest = i
    longest = line

pd.Series((matrices.mean(0)*matrices).sum(1).sum(1)/(matrices.sum(1).sum(1))).sort_values().head() 

r_s = pd.Series(np.argsort(np.argsort(np.product(np.abs((1 - matrices) - matrices.mean(0)), axis=(1,2)))))



