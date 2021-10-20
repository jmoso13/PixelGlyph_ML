from utils import get_conv_multilabel_nn, get_model_callbacks
from pixelglyph_render import render_glyph
import numpy as np
from PIL import Image
import pickle as pkl
import pandas as pd
from sklearn.metrics import roc_auc_score
import glob
import argparse
import os
import shutil

#----------------------------------------------------------------------------

# Matrices
with open('data/pixeldict.pkl','rb') as f:
  pixeldict = pkl.load(f)  

#----------------------------------------------------------------------------

def run(label_csv, small_glyph_dir, include_labels, use_matrices, return_top_n, trained_weights, convert_greyscale, num_epochs):
  # Grab Labels
  label_df = pd.read_csv(label_csv)
  all_ids = np.arange(10000) + 1
  pos_dict = dict(zip(all_ids, np.argsort(all_ids)))
  not_labels = ['id', 'image', 'link', 'name', 'other']
  labels = [c for c in label_df.columns if c not in not_labels]
  label_sub = include_labels if include_labels != 'all' else labels

  # Clip multiple votes to binary label
  # TODO - perhaps weight labels to take into account agreement
  label_df.loc[:, labels] = label_df.loc[:, labels].clip(0,1)
  Y = np.array([label_df.loc[label_df['id'] == i, label_sub].values[0] for i in all_ids])
  
  # Remove unvoted glyphs
  label_df = label_df.loc[label_df[labels].sum(1) > 0]
  labeled_ids = np.array(label_df['id'].tolist())
  train_pct = 0.75 if num_epochs > 16 else 0.94

  # Use Matrices for training
  if use_matrices:
    X = np.array([pixeldict[i][0] for i in all_ids])
    train_idx = np.random.choice(labeled_ids, int((train_pct)*len(labeled_ids)), replace=False)
    val_idx = [i for i in labeled_ids if i not in train_idx]
    test_idx = [i for i in all_ids if i not in labeled_ids]
    x_train = X[[pos_dict[i] for i in train_idx]].reshape(-1, 10, 5, 1) - 0.5
    x_val = X[[pos_dict[i] for i in val_idx]].reshape(-1, 10, 5, 1) - 0.5
    x_test = X[[pos_dict[i] for i in test_idx]].reshape(-1, 10, 5, 1) - 0.5
    y_train = Y[[pos_dict[i] for i in train_idx]]
    y_val = Y[[pos_dict[i] for i in val_idx]]
 
  # Or use raw images
  else:
    # Load
    glyph_list = list()
    for g in glob.glob(os.path.join(small_glyph_dir,'*.png')):
      glyph_list.append(np.asarray(Image.open(g)))

    X = np.array([glyph_list[i-1] for i in all_ids]).astype(float)/255.
    if convert_greyscale:
      X = np.dot(X,np.array([[0.2989], [0.5870], [0.1140]]))
    train_idx = np.random.choice(labeled_ids, int((train_pct)*len(labeled_ids)), replace=False)
    val_idx = [i for i in labeled_ids if i not in train_idx]
    test_idx = [i for i in all_ids if i not in labeled_ids]
    x_train = X[[pos_dict[i] for i in train_idx]]
    x_val = X[[pos_dict[i] for i in val_idx]]
    x_test = X[[pos_dict[i] for i in test_idx]]
    y_train = Y[[pos_dict[i] for i in train_idx]]
    y_val = Y[[pos_dict[i] for i in val_idx]]


  # CNN Training
  num_classes = Y[0].shape[0]
  str_labels = '-'.join(label_sub)
  model_name = f'pixelglyph_matrices_cnn__{str_labels}__labels' if use_matrices else f'pixelglyph_cnn__{str_labels}__labels'
  if convert_greyscale:
    model_name += '_greyscale'
  mat_shape = x_train[0].shape

  conf = {
    'num_filters':[32,64,64],
    'kernel_size':[3,2,3],
    'stride':[1,1,2],
    'batch_norm':True,
    'max_pooling':[2,2,2]
  } if not use_matrices else {
    'num_filters':[32,64,64],
    'kernel_size':[3,2,3],
    'stride':[1,1,1],
    'batch_norm':True,
    'max_pooling':[2,2,1]
  }

  nn = get_conv_multilabel_nn(mat_shape, num_classes, model_name, num_filters=conf['num_filters'], kernel_size=conf['kernel_size'], stride=conf['stride'], batch_norm=conf['batch_norm'], max_pooling=conf['max_pooling'])
  es, mc = get_model_callbacks(model_name, patience=20)

  if trained_weights is None:
    # Hyperparameters
    batch_size = 32
    epochs = num_epochs

    nn.fit(x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_val, y_val),
            callbacks=[es, mc])

    if num_epochs > 16:
      print('loading weights')
      nn.load_weights(f'models/best_{model_name}.h5')
    else:
      print('saving weights')
      nn.save_weights(f'models/best_{model_name}.h5')

  # Load model without training
  else:
    print('Skipping training...')
    nn.load_weights(trained_weights)

  if not os.path.exists('unlabeled_preds'):
    os.mkdir('unlabeled_preds')

  os.system(f"rm -rf {' '.join([os.path.join('unlabeled_preds', l) for l in label_sub])}")

  # Render and save top n predicted glyphs for each class
  for label in label_sub:
    arg_idx = np.argwhere(np.array(label_sub) == label)[0][0]
    print(f'Rendering top {return_top_n} most likely to be in {label} label')
    try:
      print(f'ROC AUC on valid set of {roc_auc_score(y_score=nn.predict(x_val)[:,arg_idx], y_true=y_val[:,arg_idx])}')
    except:
      print(f'No AUC for this label because no examples in val set')
    if not os.path.exists(f'unlabeled_preds/{label}'):
      os.mkdir(f'unlabeled_preds/{label}')
    test_preds = nn.predict(x_test)
    top_100 = np.array(list(reversed(np.argsort(test_preds[:,arg_idx])[-return_top_n:])))
    top_100_ids = np.array(test_idx)[top_100]
    for n, i in enumerate(top_100_ids):
      matrix, colors = pixeldict[i]
      render_glyph(matrix, colors, f'unlabeled_preds/{label}/glyph_n{n:05d}_{i:05d}')

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train CNN on pixelglyphs, be sure to run pixelglyph_render.py first to populate a directory for small glyphs
  python pixelglyph_supervsed.py --label-csv data/glyph_labels.csv --small-glyph-dir small_glyphs/ --use-matrices=no

  # Run rendering again without retraining (be sure to match --include-labels to appropriate model)
  python pixelglyph_supervsed.py --label-csv data/glyph_labels.csv --small-glyph-dir small_glyphs/ --use-matrices=no --trained-weights models/pixelglyph_cnn.h5

'''

def main():
  parser = argparse.ArgumentParser(
    description = 'Train Supervised PG Classifier', 
    epilog=_examples, 
    formatter_class=argparse.RawDescriptionHelpFormatter
    )
  # Adapt this to pull from database if no csv
  parser.add_argument('--label-csv', help='Location of CSV where labels can be found', required=True)
  parser.add_argument('--small-glyph-dir', help='Location of directory where small glyph renders can be found', required=True, metavar='DIR')
  parser.add_argument('--include-labels', nargs='+', help='List of labels to train for, default to all of them', default='all')
  parser.add_argument('--use-matrices', help='Whether to use pixelglyph matrices rather than raw images', default=False, metavar='BOOL', type=_str_to_bool)
  parser.add_argument('--return-top-n', help='How many of top likely images to save for each class', default=100, type=int)
  parser.add_argument('--trained-weights', help='Location of pre-trained weights, skips training and uses trained network', default=None)
  parser.add_argument('--convert-greyscale', help='Whether to convert raw images to greyscale', default=False, metavar='BOOL', type=_str_to_bool)
  parser.add_argument('--num-epochs', help='Whether to convert raw images to greyscale', default=500, type=int)
  args = parser.parse_args()

  if not os.path.exists(args.small_glyph_dir):
    print(f'--small-glyph-dir == {args.small_glyph_dir} is not a valid location')
    raise Exception

  if not os.path.exists(args.label_csv):
    print(f'--label-csv == {args.label_csv} is not a valid location')
    raise Exception 

  if args.trained_weights is not None:
    if not os.path.exists(args.trained_weights):
      print(f'--trained-weights == {args.trained_weights} is not a valid location')
      raise Exception 

  run(**vars(args))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
