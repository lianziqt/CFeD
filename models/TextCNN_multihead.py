#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MyModel import MyModel


class Config(object):

    def __init__(self, scenario, embedding):
        self.model_name = 'TextCNN_multihead'

        if 'class' in scenario:
            self.train_tasks = [
                'dataset/thucnews_train_classIL1.csv',
                'dataset/thucnews_train_classIL2.csv',
                'dataset/thucnews_train_classIL3.csv',
                'dataset/surrogate_dataset.csv',
            ]
            self.dev_tasks = [
                'dataset/thucnews_eval_classIL1.csv',
                'dataset/thucnews_eval_classIL2.csv',
                'dataset/thucnews_eval_classIL3.csv',
            ]
            self.test_tasks = [
                'dataset/thucnews_test_classIL1.csv',
                'dataset/thucnews_test_classIL2.csv',
                'dataset/thucnews_test_classIL3.csv',
            ]
            self.task_number = len(self.test_tasks)
            self.class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
            self.task_list = [
                ['0', '1', '2', ],
                ['3', '4', '5', ],
                ['6', '7', '8', ],
            ]
        else : #domain-IL
            self.train_tasks = [
                'dataset/thucnews_train_domainIL.csv',
                'dataset/sina2019_train_domainIL.csv',
                'dataset/sogou_train_domainIL.csv',
                'dataset/surrogate_dataset.csv',
            ]
            self.dev_tasks = [
                'dataset/thucnews_eval_domainIL.csv',
                'dataset/sina2019_eval_domainIL.csv',
                'dataset/sogou_eval_domainIL.csv'
            ]
            self.test_tasks = [
                'dataset/thucnews_test_domainIL.csv',
                'dataset/sina2019_test_domainIL.csv',
                'dataset/sogou_test_domainIL.csv'
            ]
            self.task_number = len(self.test_tasks)
            self.class_list = ['0', '1', '2','0', '1', '2', '0', '1', '2']
            self.task_list = [
                ['0', '1', '2', ],
                ['0', '1', '2', ],
                ['0', '1', '2', ],
            ]

        self.vocab_path = 'dataset/vocab.pkl'
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.1                                              # probability of drop out
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 20                                            # number of epoch or communication round
        # self.batch_size = 128
        self.batch_size = 4
        self.num_classes_of_task = [len(task) for task in self.task_list] 

        # self.num_users = 100
        # self.frac = 0.1
        self.num_users = 5
        self.frac = 1.0
        self.local_ep = 40
        # self.local_bs = 10
        self.local_bs = 1
        self.iid = True
        self.server_distillation = False

        self.verbose = 1
        self.pad_size = 32
        self.learning_rate = 1e-6
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2, 3, 4)                                   # kernel size of CNN
        self.num_filters = 256                                          # channels of CNN


class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), 100)
        self.fcs = nn.ModuleList(
            [nn.Linear(100, num_classes)
             for num_classes in config.num_classes_of_task
             ]
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        outs = [fc(out) for fc in self.fcs]
        return outs


if __name__ == '__main__':
    config = Config()
