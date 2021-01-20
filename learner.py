import os
import sys
import pdb
import six
import random
# import lmdb
from PIL import Image
import numpy as np
import math
from collections import OrderedDict
from itertools import chain
import logging
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import random_split
from utils import *
from trainer import OCRTrainer
from models import CRNN ,STNet
from loss import CustomCTCLoss
from dataset import SynthDataset, SynthCollator


class Learner(object):
    def __init__(self, model, optimizer, savepath=None, resume=False):
        self.model = model
        self.optimizer = optimizer
        self.savepath = os.path.join(savepath, 'best.ckpt')
        self.cuda = torch.cuda.is_available()
        self.cuda_count = torch.cuda.device_count()
        if self.cuda:
            self.model = self.model.cuda()
        self.epoch = 0
        if self.cuda_count > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.best_score = None
        if resume and os.path.exists(self.savepath):
            self.checkpoint = torch.load(self.savepath)
            self.epoch = self.checkpoint['epoch']
            self.best_score = self.checkpoint['best']
            self.load()
        else:
            print('checkpoint does not exist')

    def fit(self, opt):
        opt['cuda'] = self.cuda
        opt['model'] = self.model
        opt['optimizer'] = self.optimizer
        logging.basicConfig(filename="%s/%s.csv" %
                            (opt['log_dir'], opt['name']), level=logging.INFO)
        self.saver = EarlyStopping(
            self.savepath, patience=5, verbose=True, best_score=self.best_score)
        opt['epoch'] = self.epoch
        trainer = OCRTrainer(opt)

        for epoch in range(opt['epoch'], opt['epochs']):
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            trainer.count = epoch
            info = '%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f' % (epoch, train_result['train_loss'],
                                                               val_result['val_loss'], train_result['train_ca'],  val_result['val_ca'],
                                                               train_result['train_wa'], val_result['val_wa'])
            logging.info(info)
            self.val_loss = val_result['val_loss']
            print(self.val_loss)
            if self.savepath:
                self.save(epoch)
            if self.saver.early_stop:
                print("Early stopping")
                break

    def load(self):
        print('Loading checkpoint at {} trained for {} epochs'.format(
            self.savepath, self.checkpoint['epoch']))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        if 'opt_state_dict' in self.checkpoint.keys():
            print('Loading optimizer')
            self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])

    def save(self, epoch):
        self.saver(self.val_loss, epoch, self.model, self.optimizer)


if __name__ == "__main__":
    alphabet = """*QWERTYUIOPASDFGHJKLZXCVBNM1234567890-"""
    args = {
        'name': 'exp3',
        'path': 'content',
        'imgdir': 'train',
        'imgH': 32,
        'nChannels': 1,
        'nHidden': 256,
        'nClasses': len(alphabet)+1,
        'lr': 0.001,
        'epochs': 70,
        'batch_size': 16,
        'save_dir': 'checkpoints',
        'log_dir': 'logs',
        'resume': False,
        'cuda': True,
        'schedule': False

    }
    data = SynthDataset(args)
    args['collate_fn'] = SynthCollator()
    train_split = int(0.8*len(data))
    val_split = len(data) - train_split
    args['data_train'], args['data_val'] = random_split(
        data, (train_split, val_split))
    print('Traininig Data Size:{}\nVal Data Size:{}'.format(
        len(args['data_train']), len(args['data_val'])))
    args['alphabet'] = alphabet
    model = CRNN(args)
    # model =  nn.Sequential(STNet(),
    #                     CRNN(args))
    args['criterion'] = CustomCTCLoss()
    savepath = os.path.join(args['save_dir'], args['name'])
    gmkdir(savepath)
    gmkdir(args['log_dir'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    learner = Learner(model, optimizer, savepath=savepath,
                      resume=args['resume'])
    learner.fit(args)
