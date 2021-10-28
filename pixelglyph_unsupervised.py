from utils import get_conv_vae_dict, build_decoder, project, get_model_callbacks, save_random_pred, project_matrix
from pixelglyph_render import render_glyph
from keras.layers import Input
import numpy as np
import pandas as pd
import os
import glob
import pickle as pkl
from PIL import Image

#----------------------------------------------------------------------------
### WIP NOT FINISHED ###
# Matrices
with open('data/pixeldict.pkl','rb') as f:
  pixeldict = pkl.load(f)  

#----------------------------------------------------------------------------

def run(small_glyph_dir, use_matrices, trained_weights, convert_greyscale, loss_weights, num_epochs):
  all_ids = np.arange(10000) + 1
  pos_dict = dict(zip(all_ids, np.argsort(all_ids)))
  train_pct = 0.75

    # Use Matrices for training
  if use_matrices:
    X = np.array([pixeldict[i][0] for i in all_ids])
    matrix_means = X.mean(0)
    train_idx = np.random.choice(all_ids, int((train_pct)*len(all_ids)), replace=False)
    val_idx = [i for i in all_ids if i not in train_idx]
    x_train = X[[pos_dict[i] for i in train_idx]].reshape(-1, 10, 5, 1) - 0.5
    x_val = X[[pos_dict[i] for i in val_idx]].reshape(-1, 10, 5, 1) - 0.5
 
  # Or use raw images
  else:
    # Load
    glyph_list = list()
    for g in glob.glob(os.path.join(small_glyph_dir,'*.png')):
      glyph_list.append(np.asarray(Image.open(g)))

    X = np.array([glyph_list[i-1] for i in all_ids]).astype(float)/255.
    if convert_greyscale:
      X = np.dot(X,np.array([[0.2989], [0.5870], [0.1140]]))

    train_idx = np.random.choice(all_ids, int((train_pct)*len(all_ids)), replace=False)
    val_idx = [i for i in all_ids if i not in train_idx]
    x_train = X[[pos_dict[i] for i in train_idx]]
    x_val = X[[pos_dict[i] for i in val_idx]]

  #VAE
  lowest_dim = 2
  model_name = 'pixelglyph_vae'
  train_shape = x_train[0].shape
  channels = train_shape[-1]

  conf = {
    'e_f':[32,64,64,64],
    'e_ks':[3,3,3,3],
    'e_s':[2,2,2,2],
    'd_f':[64,64,32,channels],
    'd_ks':[3,3,3,3],
    'd_s':[2,2,2,2],
    'batch_norm':True,
    'dropout_rate':0.0625
  } if not use_matrices else {
    'e_f':[32,20,8], 
    'e_ks':[3,2,3], 
    'e_s':[1,1,1], 
    'd_f':[8,20,32,1], 
    'd_ks':[3,2,3,3], 
    'd_s':[1,1,1,1],
    'batch_norm':True,
    'dropout_rate':0.0
  }

  vae_dict = get_conv_vae_dict(train_shape, lowest_dim, model_name, loss_weights=loss_weights, e_f=conf['e_f'], e_ks=conf['e_ks'], e_s=conf['e_s'], d_f=conf['d_f'], d_ks=conf['d_ks'], d_s=conf['d_s'], batch_norm=conf['batch_norm'], dropout_rate=conf['dropout_rate'])
  vae = vae_dict['vae']
  encoder = vae_dict['encoder']
  get_losses = vae_dict['get_losses_function']
  es, mc = get_model_callbacks(model_name, patience=20)

  # Hyperparameters
  batch_size = 32
  epochs = num_epochs

  vae.fit(x_train,
          epochs=epochs,
          batch_size=batch_size,
          shuffle=True,
          validation_data=(x_val, None),
          callbacks=[es, mc])

  vae.load_weights(f'models/best_{model_name}.h5')

  # Sample
  save_random_pred(vae, x_val, model_name)

  decoder_input = Input(shape=(2,)) 
  decoder = build_decoder(vae, decoder_input)
  points = np.linspace(-3, 3, 30)
  if use_matrices:
    ppd = 10
    g_dim = ppd*16
    full_pic = np.zeros((g_dim*len(points), g_dim*len(points), 3))
  else:
    full_pic = np.zeros((train_shape[1]*len(points), train_shape[0]*len(points), channels))

  for i, y in enumerate(points):
    for j, x in enumerate(points):
      print(x,y)
      if use_matrices:
        mat = project_matrix(decoder, [x,y], matrix_means=matrix_means, rounding=True)
        colors = ['rgb(203,201,208)', 'rgb(166,170,189)', 'rgb(95,98,114)']
        num = render_glyph(mat.reshape(10,5), colors, 'test_colors', ppd=ppd)
      else:
        num = project(decoder, [x,y])
      x_dim = num.shape[1]
      y_dim = num.shape[0]
      full_pic[y_dim*len(points)-(i*y_dim+y_dim):y_dim*len(points)-i*y_dim, j*x_dim:j*x_dim+x_dim] = num

  if convert_greyscale:
    im = Image.fromarray(np.squeeze(full_pic.astype('uint8'),axis=-1), 'L')
  else:
    im = Image.fromarray(full_pic.astype('uint8'))
  im.save(f'{model_name}_latent_space.png')



if not os.path.exists('reel/'):
  os.mkdir('reel/')

x, y = np.random.uniform(-4,4), np.random.uniform(-4,4)
x_end, y_end = np.random.uniform(-4,4), np.random.uniform(-4,4)

fps = 59.94
secs = 12
frames = fps*secs
distance = np.sqrt(np.square(x_end-x) + np.square(y_end-y))

print(distance)

m = (y_end-y)/(x_end-x)
b = y - m*x

for frame in range(int(np.ceil(frames))):
  new_x = x + (frame/frames)*(x_end-x)
  new_y = m*new_x + b
  print(round(np.sqrt(np.square(new_x-x) + np.square(new_y-y))/distance, 4))
  mat = project_matrix(decoder, [new_x,new_y], matrix_means=matrix_means, rounding=True)
  colors = ['rgb(203,201,208)', 'rgb(166,170,189)', 'rgb(95,98,114)']
  image = render_glyph(mat, colors, ppd=37)
  Image.fromarray(image).save(f'reel/reel_{frame:09d}.png')

os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i 'reel/*.png' -f mov -pix_fmt yuv420p glyph_test.mov")
os.system(f'rm reel/*.png')














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



