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
