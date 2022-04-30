# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.MyModel import MyModel


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'CNN_Cifar10'
        self.train_path = dataset + '/data/train_012.csv'                                # 训练集
        self.dev_path = dataset + '/data/eval_012.csv'                                   # 验证集
        self.test_path = dataset + '/data/test_012.csv'                                  # 测试集

        self.train_tasks = [
            'dataset/cifar_train_012',
            'dataset/cifar_train_345',
            'dataset/cifar_train_6789',

            # dataset + '/data/cifar_train_0123',
            # dataset + '/data/cifar_train_4567',
            # dataset + '/data/cifar_train_89',

            'dataset/cifar100_0',
            'dataset/caltech256',
        ]
        self.dev_tasks = [
            'dataset/cifar_eval_012',
            'dataset/cifar_eval_345',
            'dataset/cifar_eval_6789',
            # '/data/cifar_eval_0123',
            # '/data/cifar_eval_4567',
            # '/data/cifar_eval_89',

        ]
        self.test_tasks = [
            'dataset/cifar_eval_012',
            'dataset/cifar_eval_345',
            'dataset/cifar_eval_6789',
            # dataset + '/data/cifar_eval_0123',
            # dataset + '/data/cifar_eval_4567',
            # dataset + '/data/cifar_eval_89',

        ]

        self.task_number = 3

        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单
        # self.class_list = ['0', '1', '2']
        self.class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.task_list = [
            ['0', '1', '2', ],
            ['3', '4', '5', ],
            ['6', '7', '8', '9'],
            #
            # ['0', '1', '2', '3', ],
            # ['4', '5', '6', '7', ],
            # ['8', '9'],

        ]

        # self.class_list_tasks = [
        #     ['0', '1', '2'],
        #     ['3', '4', '5'],
        # ]


        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单

        # self.class_list_tasks = [
        #     ['0', '1', '2'],
        #     ['3', '4', '5'],
        # ]
        self.vocab_path = 'dataset/vocab.pkl'
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'        # save path of trained model
        self.log_path = 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('dataset/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # pre-trained word vector
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # GPU

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        # self.num_classes = 6
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                           # epoch数
        self.batch_size = 512                                          # mini-batch大小
        self.schedule_gamma = 0.1

        self.num_users = 100
        self.frac = 0.1
        self.local_ep = 10
        self.local_bs = 5
        self.iid = True
        self.server_distillation = False

        self.verbose = 1
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-3                                     # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''

# class Model(MyModel):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.conv1_out_channels = 64
#         self.conv2_out_channels = 128
#         self.conv1 = nn.Conv2d(3, self.conv1_out_channels, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, 5)
#         self.dropout = nn.Dropout(args.dropout)
#         self.batchnorm1 = nn.BatchNorm2d(self.conv1_out_channels)
#         self.batchnorm2 = nn.BatchNorm2d(self.conv2_out_channels)
#         self.fc1 = nn.Linear(self.conv2_out_channels * 5 * 5, 100)
#         # self.fc2 = nn.Linear(500, 50)
#         # self.fc3 = nn.Linear(100, args.num_classes)
#         self.fcs = nn.ModuleList(
#             [nn.Linear(100, len(classes))
#              for classes in args.task_list
#              ]
#         )

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.batchnorm1(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.batchnorm2(x)
#         x = x.view(-1, self.conv2_out_channels * 5 * 5)
#         # x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))

#         # x = self.fc3(x)
#         outs = [fc(x) for fc in self.fcs]
#         return outs
#         # return F.log_softmax(x, dim=1)
#         # return x


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
        self.dropout = nn.Dropout(0.2)

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
        # self.fc = nn.Linear(512, config.num_classes)
        self.fcs = nn.ModuleList(
            [nn.Linear(512, len(classes))
             for classes in config.task_list
             ]
        )

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
        # out = self.fc(out)
        outs = [fc(out) for fc in self.fcs]
        return outs


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#         )
#         self.batchnorm = nn.BatchNorm2d(out_channels)
#         self.left2 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#             )
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         out = self.left(x)
#         out = self.batchnorm(out)
#         out = self.left2(out)
#         out = self.batchnorm(out)
#         out += self.batchnorm(self.shortcut(x))
#         # out = self.batchnorm(out)
#         out = F.relu(out)
#         out = self.dropout(out)
#         return out


# class Model(MyModel):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#         )
#         self.batchnorm = nn.BatchNorm2d(64)
#         self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.fc = nn.Linear(512, config.num_classes)

#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.batchnorm(out)
#         out = F.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out