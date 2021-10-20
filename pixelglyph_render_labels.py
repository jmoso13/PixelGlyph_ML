from pixelglyph_render import render_glyph
import numpy as np
from PIL import Image
import pickle as pkl
import re
import os
import argparse
import pandas as pd

#----------------------------------------------------------------------------

def main():
  label_df = pd.read_csv('data/glyph_labels.csv')

  with open('data/pixeldict.pkl','rb') as f:
    pixeldict = pkl.load(f)

  if not os.path.exists('label_investigation'):
    os.mkdir('label_investigation')

  for label in labels:
    print(label)
    os.mkdir(f'label_investigation/{label}')
    render_df = label_df.loc[label_df[label] > 0, ['id', label]].sort_values(label,ascending=False)
    for i, l in render_df.values:
      matrix, colors = pixeldict[i]
      render_glyph(matrix, colors, f'label_investigation/{label}/{l:03d}_votes_{i:05d}.png', ppd=37)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------