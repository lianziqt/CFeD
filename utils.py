# coding: UTF-8
import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from PIL import Image
import copy
import math
# import cv2 as cv


MAX_VOCAB_SIZE = 10000  
UNK, PAD = '<UNK>', '<PAD>'  


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        # print('Name: ', name)
        if exclude in name:
            continue
        if 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            if 'batch' in name:
                nn.init.normal_(w)
                continue
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)

def init_network_resnet(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None: nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal_(m.weight.data, mean=0., std=math.sqrt(2. / fan_in))
            if m.bias is not None: nn.init.zeros_(m.bias.data)
            
            
def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # word-level, split the word with space
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print('Vocab size: {}'.format(len(vocab)))

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label)))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


def build_dataset_cifar10(config):
    def load_dataset_from_pkl(path):
        with open(path, 'rb') as f:
            dataset = pkl.load(f, encoding='bytes')
        
        contents = []
        for line in tqdm(dataset):
            img = np.array(line[0])
            if(img.shape != (3, 32, 32)):
                img = img.reshape(3, 32, 32)
            # img = img.transpose((1, 2, 0))
            # img = Image.fromarray(img)
            # print(img.shape)
            contents.append((img, int(line[1])))
            
            # print(line[1])
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]
    vocab = {}
    # train = load_dataset_from_csv(config.train_path, config.pad_size)
    trains = [load_dataset_from_pkl(train_path) for train_path in config.train_tasks]
    devs = [load_dataset_from_pkl(dev_path) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_pkl(test_path) for test_path in config.test_tasks]
    return vocab, trains, devs, tests

def build_dataset_cifar100(config):
    def load_dataset_from_pkl(path):
        with open(path, 'rb') as f:
            dataset = pkl.load(f, encoding='bytes')
        
        contents = []
        for line in tqdm(dataset):
            img = np.array(line[0])
            if(img.shape != (3, 32, 32)):
                img = img.reshape(3, 32, 32)
            # img = img.transpose((1, 2, 0))
            # img = Image.fromarray(img)
            # print(img.shape)
            contents.append((img, int(line[1])))
            
            # print(line[1])
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]
    trains = []
    evals = []
    tests = []
    with open(config.label2data_train_path, 'rb') as f1:
        with open(config.label2data_eval_path, 'rb') as f2:
            with open(config.label2data_test_path, 'rb') as f3:
                dataset = []
                dataset.append(pkl.load(f1, encoding='bytes'))
                dataset.append(pkl.load(f2, encoding='bytes'))
                dataset.append(pkl.load(f3, encoding='bytes'))
                tasks = construct_task_seq_cifar(dataset, task_size=config.task_size, seed=config.seed)
                trains.extend(tasks[0])
                evals.extend(tasks[1])
                tests.extend(tasks[2])
    trains.append(load_dataset_from_pkl(config.train_tasks[-1]))
            
    vocab = {}
    return vocab, trains, evals, tests

def build_dataset_from_csv(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # word-level, split the word with space
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab_from_csv(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print('Vocab size: {}'.format(len(vocab)))

    def load_dataset_from_csv(path, pad_size=32):
        dataset = pd.read_csv(path)
        dataset = dataset.values
        contents = []
        for line in tqdm(dataset):
            try:
                content, label = str(line[0]).strip() + str(line[1]).strip(), int(line[2])
            except AttributeError:
                print(content, label)
                content, label = line[0].strip(), int(line[2])

            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label)))
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]
    trains = [load_dataset_from_csv(train_path, config.pad_size) for train_path in config.train_tasks]
    devs = [load_dataset_from_csv(dev_path, config.pad_size) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_csv(test_path, config.pad_size) for test_path in config.test_tasks]
    return vocab, trains, devs, tests


def build_dataset_from_csv_fed(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # word-level, split the word with space
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab_from_csv(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print('Vocab size: {}'.format(len(vocab)))

    def load_dataset_from_csv(path, pad_size=32):
        print(path)
        dataset = pd.read_csv(path)
        dataset = dataset.values
        contents = []
        for line in tqdm(dataset):
            try:
                content, label = str(line[-3]).strip() + str(line[-2]).strip(), int(line[-1])
            except:
                content, label = line[1].strip(), int(line[2])
                print(content, label)

            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label)))
        np.random.shuffle(contents)
        return contents  # [([...], 0), ([...], 1), ...]

    trains = [load_dataset_from_csv(train_path, config.pad_size) for train_path in config.train_tasks]
    devs = [load_dataset_from_csv(dev_path, config.pad_size) for dev_path in config.dev_tasks]
    tests = [load_dataset_from_csv(test_path, config.pad_size) for test_path in config.test_tasks]
    return vocab, trains, devs, tests


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


class CifarIterater(DatasetIterater):
    def __init__(self, batches, batch_size, device):
        super(CifarIterater, self).__init__(batches, batch_size, device)

    def _to_tensor(self, datas):
        x = torch.FloatTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return x, y


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def build_cifar_iterator(dataset, config):
    iter = CifarIterater(dataset, config.batch_size, config.device)
    return iter


def build_usergroup(dataset, config):
    num_shards, num_texts = 200, len(dataset) // 200
    num_assign = num_shards // config.num_users
    # print(num_shards)
    # print(num_texts)
    # print(num_assign)

    # print(len(dataset))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(config.num_users)}
    idxs = np.arange((num_shards - 1) * num_texts)

    # divide and assign 2 shards/client
    for i in range(config.num_users):
        rand_set = set(np.random.choice(idx_shard, num_assign, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_texts:(rand + 1) * num_texts]), axis=0)
    return dict_users


def build_usergroup_non_iid(dataset, config):
    num_shards, num_texts = 200, len(dataset) // 200
    print(len(dataset))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(config.num_users)}
    idxs = np.arange((num_shards - 1) * num_texts)
    # labels = dataset.train_labels.numpy()

    labels = np.asarray([content[1] for content in dataset], dtype=np.float64)

    # sort labels
    idxs_labels = np.vstack((idxs, labels[:(num_shards - 1) * num_texts]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(config.num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_texts:(rand + 1) * num_texts]), axis=0)
    return dict_users

def build_usergroup_full_dataset(dataset, config):
    dict_users = {i: np.array([]) for i in range(config.num_users)}
    idxs = np.arange(len(dataset))

    # divide and assign 2 shards/client
    for i in range(config.num_users):
            dict_users[i] = np.concatenate((dict_users[i], idxs), axis=0)
    return dict_users

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def _update_mean_params(model):
    for param_name, param in model.named_parameters():
        _buff_param_name = param_name.replace('.', '_')
        model.register_buffer(_buff_param_name, param.data.clone())

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # print(key)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print('    Learning  : {}'.format(args.learning_rate))
    print('    Global Rounds   : {}\n'.format(args.num_epochs))

    print('    Federated parameters:')
    print('    Fraction of users  : {}'.format(args.frac))
    print('    Local Batch size   : {}'.format(args.local_bs))
    print('    Local Epochs       : {}\n'.format(args.local_ep))


def construct_task_seq(label2data, data_func, task_size, seed, rand_seq):
    num_classes = len(label2data)
    np.random.seed(seed)
    tasks = []

    for i in range(0, num_classes, task_size):
        task = []
        for step in range(task_size):
            cur_class = rand_seq[i + step]
            cur_class_data = label2data[cur_class]    
            task.extend([(data_func(d), cur_class) for d in cur_class_data])
        np.random.shuffle(task)
        np.random.shuffle(task)
        tasks.append(task)
    return tasks
    
def construct_task_seq_cifar(label2datas, task_size=5, seed=1):
    if type(label2datas) is list:
        num_classes = len(label2datas[0])
    else:
        num_classes = len(label2datas)

    rand_seq = list(np.random.permutation(num_classes))
    
    def data_func(data):
        img = np.array(data)
        if(img.shape != (3, 32, 32)):
            img = img.reshape(3, 32, 32)
        return img
    if type(label2datas) is list:
        return [construct_task_seq(l, data_func, task_size, seed, rand_seq) for l in label2datas]
    else:
        return construct_task_seq(label2datas, data_func, task_size, seed, rand_seq)