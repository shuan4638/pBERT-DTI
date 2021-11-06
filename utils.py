import torch
import sklearn
import errno
import json
import os
import numpy as np
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from PAL import PairwiseAttentionLinear

class Subset(object):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, item):
        return self.dataset[self.indices[item]]
    def __len__(self):
        return len(self.indices)

class DTIDataset(object):
    def __init__(self, dataset_name, log_every=1000):
        df = pd.read_csv('data/%s_data.csv' % dataset_name)
        self.drugs = df['Drug'].values
        self.targets = df['Target'].values
        self.labels = df['Y']
        self.train_ids = df.index[df['Split'] == 'train'].values
        self.val_ids = df.index[df['Split'] == 'valid'].values
        self.test_ids = df.index[df['Split'] == 'test'].values
        # self.drug_feats = np.load('data/drug_fps.npy', allow_pickle = 'TRUE').item()
        self.drug_feats = np.load('data/drug_features.npy', allow_pickle = 'TRUE').item()
        self.prot_feats = np.load('data/seq_features.npy', allow_pickle = 'TRUE').item()

    def __getitem__(self, item):
        return self.drug_feats[self.drugs[item]], self.prot_feats[self.targets[item]], int(self.labels[item])

    def __len__(self):
        return len(self.drugs)
			
class EarlyStopping(object):
    def __init__(self, mode='higher', patience=10, filename=None, metric='pr_auc_score'):
        self.mode = mode
        self._check = self._check_higher
        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


    def save_checkpoint(self, model):
        torch.save({'model_state_dict': model.state_dict(),
                    'timestep': self.timestep}, self.filename)


    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])
		
	
def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory %s'% path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory %s already exists.' % path)
        else:
            raise

def load_dataloader(args):
	dataset = DTIDataset(args['dataset'])
	train_set, val_set, test_set = Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids), Subset(dataset, dataset.test_ids)
	train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True)
	val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'])
	test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'])
	return train_loader, val_loader, test_loader

def load_model(args):
    model = PairwiseAttentionLinear(n_layers = args['n_layers'])
    model = model.to(args['device'])
    loss_criterion = torch.nn.BCEWithLogitsLoss()
    # loss_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
    return model, loss_criterion, optimizer, stopper

