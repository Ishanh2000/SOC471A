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
from social_lstm_model import train, SocialLSTM
import social_lstm_model
import sna

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--log_file", type=str, default=None, 
            help="Where to log the model training details.")
    parser.add_argument("--save_embeds", action='store_true',
            help="Whether to save the hidden-state LSTM embeddings that are generated.\
                  They will be stored based on the log_file name used above.")
    parser.add_argument("--dropout", type=float, default=0.2,
            help="Dropout rate for inter-LSTM layers in 2-layer LSTM.")
    parser.add_argument("--single_layer", action='store_true',
            help="Use single-layer LSTM (implies that dropout param is ignored)")
    parser.add_argument("--include_meta", action='store_true',
            help="Include metadata/hand-crafted features in final layer of model.")
    parser.add_argument("--final_dense", action='store_true',
            help="Include an extra Linear+ReLU layer before the softmax.")
    parser.add_argument("--lstm_append_social", action='store_true', 
            help="Append the social embeddings instead of prepending them to LSTM input.")
    parser.add_argument("--lstm_no_social", action='store_true', 
            help="Do not include social embeddings in LSTM input.")
    parser.add_argument("--final_layer_social", action='store_true', 
            help="(Also) include social embeddings in the final layer.")
    args = parser.parse_args()
    dropout = None if args.single_layer else args.dropout
    if args.lstm_append_social and args.lstm_no_social:
        raise Exception("Only one of --lstm_append_social and --lstm_no_social can be True at a time.")
    if args.log_file is None and args.save_embeds:
        raise Exception("A log file must be specified if you want to store the LSTM embeddings of the posts.")
    if args.lstm_append_social or args.lstm_no_social:
        prepend_social = None if args.lstm_no_social else False
    else:
        prepend_social = True


    print ("Loading training data")
    # WE HAVE PRE-CONSTRUCTED TRAIN/VAL/TEST DATA USING load_data
    # this avoids re-doing all the pre-processing everytime the code is
    # run. This data is fixed to a batch size of 512.
    train_data = pickle.load(open(constants.TRAIN_DATA, 'rb'))
    val_data = pickle.load(open(constants.VAL_DATA, 'rb'))
    test_data = pickle.load(open(constants.TEST_DATA, 'rb'))

    print (len(train_data)*constants.BATCH_SIZE, "training examples", len(val_data)*512, "validation examples")
    print (sum([i for batch in train_data for i in batch[-1]]), "positive training", sum([i for batch in val_data for i in batch[-1]]), "positive validation")

    # annoying checks for CUDA switches....
    if constants.CUDA:
        for i in range(len(train_data)):
            batch = train_data[i]
            metafeats = batch[5]
            train_data[i] = (batch[0], 
                    batch[1].cuda(),
                    batch[2].cuda(),
                    batch[3].cuda(),
                    batch[4],
                    metafeats.cuda(),
                    batch[6].cuda())

        for i in range(len(val_data)):
            batch = val_data[i]
            metafeats = batch[5]
            val_data[i] = (batch[0], 
                    batch[1].cuda(),
                    batch[2].cuda(),
                    batch[3].cuda(),
                    batch[4],
                    metafeats.cuda(),
                    batch[6].cuda())

        for i in range(len(test_data)):
            batch = test_data[i]
            metafeats = batch[5]
            test_data[i] = (batch[0], 
                    batch[1].cuda(),
                    batch[2].cuda(),
                    batch[3].cuda(),
                    batch[4],
                    metafeats.cuda(),
                    batch[6].cuda())

    best_auc = (0,"") 
    social_lstm_model.model = SocialLSTM(args.hidden_dim, prepend_social=prepend_social, dropout=args.dropout, include_embeds=args.final_layer_social, 
            include_meta=args.include_meta, final_dense=args.final_dense)
    if constants.CUDA:
        social_lstm_model.model.cuda()
    optimizer = torch.optim.Adam(ifilter(lambda p : p.requires_grad, social_lstm_model.model.parameters()), lr=args.learning_rate)
    auc = train(social_lstm_model.model, train_data, val_data, test_data, optimizer, epochs=10, log_file=args.log_file, save_embeds=args.save_embeds)

    sna.saveImgs()
    
