#!/usr/bin/env python

"""Train a Fast R-CNN network on a region of interest of Distracted Driver Detection database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys


## GLOBAL VARIABLES ##

SOLVER = "models/pascal_voc/VGG16/faster_rcnn_end2end/solver.prototxt"
MAX_ITERS = 40000
PRETRAINED_WEIGHTS=None
CFG_FILE = None
IMDB_NAME = 'sf_ddd_small_trainval'
RANDOMIZE = False
SET_CFGS = ['TRAIN.USE_FLIPPED', 'False']

######################

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':

    if CFG_FILE is not None:
        cfg_from_file(CFG_FILE)
    if SET_CFGS is not None:
        cfg_from_list(SET_CFGS)

    print('Using config:')
    pprint.pprint(cfg)

    if not RANDOMIZE:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_cpu()

    imdb, roidb = combined_roidb(IMDB_NAME)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # train_net(SOLVER, roidb, output_dir,
    #           pretrained_model=PRETRAINED_WEIGHTS,
    #           max_iters=MAX_ITERS)
