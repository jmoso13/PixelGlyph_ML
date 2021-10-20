import numpy as np
from PIL import Image
import pickle as pkl
import re
import os
import argparse

#----------------------------------------------------------------------------

with open('pixeldict.pkl','rb') as f:
  pixeldict = pkl.load(f)

#----------------------------------------------------------------------------

def render_glyph(matrix, colors, loc, ppd=37):

  def strip_color(color):
    r,g,b = re.findall("([0-9]+|[A-Z])", color)  
    return np.array([int(c) for c in (r,g,b)]).astype('uint8')

  def count_neighbors(matrix, i, c):
    def _get_neighbor(i_n, c_n, matrix=matrix):
      if i_n >= 0 and c_n >= 0:
        try:
          neighbor = matrix[i_n][c_n]
        except IndexError as I:
          neighbor = 0
      else:
        neighbor = 0
      return neighbor
    return _get_neighbor(i+1, c) + _get_neighbor(i-1, c) + _get_neighbor(i, c+1) + _get_neighbor(i, c-1)

  fill_color = strip_color(colors[0])
  border_color = strip_color(colors[1])
  background_color = strip_color(colors[2])

  img = np.zeros((16*ppd,16*ppd,3)).astype('uint8')
  margin = 3
  glyph = np.zeros(((16-2*margin)*ppd,(16-2*margin)*ppd,3)).astype('uint8')

  img[:,:] = background_color

  for i, row in enumerate(matrix):
    for c, col in enumerate(row):
      if col == 1:
        fill_style = fill_color
      elif count_neighbors(matrix, i, c) > 0:
        fill_style = border_color
      else: 
        fill_style = background_color

      glyph[i*ppd:(i+1)*ppd, c*ppd:(c+1)*ppd] = fill_style
      glyph[i*ppd:(i+1)*ppd, (10 - c - 1)*ppd:(10-c)*ppd] = fill_style

  img[3*ppd:13*ppd, 3*ppd:13*ppd] = glyph

  Image.fromarray(img, 'RGB').save(f'{loc}.png')

#----------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(
    description = 'Render Pixelglyphs at any given size, OGs are 37',
    formatter_class=argparse.RawDescriptionHelpFormatter
    )
  parser.add_argument('--small-glyph-dir', help='Location of directory where small glyph renders can be found', required=True, metavar='DIR')
  parser.add_argument('--ppd', help='How big in pixels to make each feature', default=1, type=int)
  args = parser.parse_args()

  if os.path.exists(args.small_glyph_dir):
    os.system(f'rm -rf {args.small_glyph_dir}')
    os.mkdir(f'{args.small_glyph_dir}')

  for i in range(10000):
    matrix, colors = pixeldict[i+1]
    render_glyph(matrix, colors, os.path.join(args.small_glyph_dir, f'glyph_{i+1:05d}'), ppd=args.ppd)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

