# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MyModel import MyModel


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'CNN_Cifar100'
        self.label2data_train_path = 'dataset/cifar-100-python/label2data_train_toy'
        self.label2data_eval_path = 'dataset/cifar-100-python/label2data_train_toy'
        self.label2data_test_path = 'dataset/cifar-100-python/label2data_test_toy'
        
        self.train_tasks = [
            'dataset/caltech256_toy', # surrogate dataset
        ]
        self.dev_tasks = [
            # 'dataset/cifar_eval_012',
            # 'dataset/cifar_eval_345',
            # 'dataset/cifar_eval_6789',

        ]
        self.test_tasks = [
            # 'dataset/cifar_eval_012',
            # 'dataset/cifar_eval_345',
            # 'dataset/cifar_eval_6789',
        ]

        
        self.task_size = 20
        self.task_number = 100 // self.task_size
        self.class_list = [str(i) for i in range(100)]        
        

        self.vocab_path = 'dataset/vocab.pkl'
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'        # save path of trained model
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # pre-trained word vector
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # GPU

        
        self.dropout = 0.2                                              
        self.require_improvement = 1000                               
        self.num_classes = len(self.class_list)                       
        self.n_vocab = 0                                              
        self.num_epochs = 40                                          
        self.batch_size = 512                                         
        self.schedule_gamma = 0.1

        self.num_users = 20
        self.frac = 0.5
        self.local_ep = 5
        self.local_bs = 10
        self.iid = True
        self.server_distillation = False

        self.learning_rate = 5e-5                                   
        self.filter_sizes = (2, 3, 4)                               
        self.num_filters = 256                                       


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, config.num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
