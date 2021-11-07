# AUM SHREEGANESHAAYA NAMAH|| AUM SHREEHANUMATE NAMAH||
from PIL import Image
import numpy as np

# all = [ 'after_effecs' ]
all = [ 'after_effecs', 'comments_rise', 'loss', 'pagerank', 'phases', 'arch', 'communities', 'model', 'percent' ]

for name in all:
  print(f"Doing for {name}.png")
  img = Image.open(f'/home/ishanhmisra/AUM/SOC471A_Project/img/{name}.png')
  arr = np.asarray(img)
  img.close()

  f = open(f'/home/ishanhmisra/AUM/SOC471A_Project/img/{name}.csv', "w")
  for row in arr:
    for i, col in enumerate(row):
      if i != 0: f.write(",")
      f.write(f"{(col[0]/255) - 0.5},{(col[1]/255) - 0.5},{(col[2]/255) - 0.5}")
    f.write("\n")
  f.close()

# AUM SHREEGANESHAAYA NAMAH|| AUM SHREEHANUMATE NAMAH||
import torch
import random
import argparse
import _pickle as pickle
import torch.nn as nn
import numpy as np

ifilter = filter
# from itertools import ifilter
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

import constants
from embeddings import Embeddings


def _load_glove_embeddings():
    print ("Loading word embeddings...")
    with open("./aaa.txt") as fp:
        embeddings = np.empty((4, constants.WORD_EMBED_DIM), dtype=np.float32)
        for i, line in enumerate(fp):
            # print(i, f"\"{line}\"")
            # embeddings[i,:] = map(float, line.split()[1:])
            embeddings[i,:] = [float(x) for x in line.split()[1:]]
    return embeddings

if __name__ == "__main__":
  print(_load_glove_embeddings())
