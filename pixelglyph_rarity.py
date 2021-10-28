import numpy as np
import pickle as pkl

class RarityTool:


  def __init__(self, pixeldict_loc):
    with open(pixeldict_loc, 'rb') as f:
      self.pixeldict = pkl.load(f)
    self.all_ids = np.arange(10000) + 1
    self.pos_dict = dict(zip(self.all_ids, np.argsort(self.all_ids)))
    self.rev_pos_dict = dict(zip(np.argsort(self.all_ids), self.all_ids))
    self.matrices = np.array([pixeldict[i][0] for i in self.all_ids])  
    self.likelihood = np.product(np.abs((1 - self.matrices) - self.matrices.mean(0)), axis=(1,2))
    self.likelihood_dict = {i:likelihood[self.pos_dict[i]] for i in self.all_ids}
    self.rank = np.argsort(np.argsort(self.likelihood))
    self.rank_dict = {i:rank[self.pos_dict[i]] for i in self.all_ids}
    self.rarity_list = [self.rev_pos_dict[i] for i in np.argsort(self.likelihood)]
    self.f_matrices = self.matrices.reshape(self.matrices.shape[0],-1)
    self.pos_indcs = [np.array([0 if n != i else 1 for n in range(self.f_matrices.shape[-1])]) for i in range(self.f_matrices.shape[-1])]
    self.selected_pixels = list()
    self.possible_pos = np.array([*map(self.pos_dict.get, self.all_ids)])


  def select_pixel(self, pixel):
    self.possible_pos = np.argwhere(self.f_matrices[self.possible_pos].dot(self.pos_indcs[pixel])).flatten()
    self.selected_pixels.append(pixel)
    
    print(f'selected pixels: {self.selected_pixels}', f'{self.possible_pos.shape[0]} out of 10000 pixelglyphs have these pixels selected')
    self.render_current_pattern()
    self.render_most_likely_glyphs(2)


  def deselect_pixel(self, pixel):
    self.selected_pixels = [sp for sp in self.selected_pixels if sp != pixel]
    
    set_list = list()
    if self.selected_pixels:
      for sp in self.selected_pixels:
        set_pos = np.argwhere(self.f_matrices.dot(self.pos_indcs[sp])).flatten().tolist()
        set_list.append(set(set_pos))
      self.possible_pos = np.array(list(set.intersection(*set_list)))
    
    else:
      self.possible_pos = np.array([*map(self.pos_dict.get, self.all_ids)])

    print(f'selected pixels: {self.selected_pixels}', f'{self.possible_pos.shape[0]} out of 10000 pixelglyphs have these pixels selected')
    self.render_current_pattern()
    self.render_most_likely_glyphs(2)


  def render_current_pattern(self):
    render = np.zeros(self.f_matrices.shape[-1])
    render[self.selected_pixels] = 1
    render = render.reshape(self.matrices.shape[1:])
    # render = FLIP THIS
    print(render)


  def render_most_likely_glyphs(self,num):
    pass


