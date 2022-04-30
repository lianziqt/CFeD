# coding: UTF-8
import time
import datetime
import torch
import numpy as np
from train_eval import train, init_network, init_network_resnet, test, train_multihead, train_CFeD_c, train_ewc, train_dmc, train_lwf

from importlib import import_module
import argparse
import copy
from utils import build_dataset, build_iterator, get_time_dif, build_dataset_from_csv, build_dataset_cifar10, \
    build_cifar_iterator, build_dataset_cifar100


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextCNN_Multihead')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--paradigm', default='', type=str, help='choose the training paradigm')
parser.add_argument('--scenario', default='domain', type=str, help=':Class-IL or Domain-IL')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  

    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    
    x = import_module('models.' + model_name)
    config = x.Config(args.scenario, embedding)
    config.seed = 999
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)    
    torch.backends.cudnn.deterministic = True  

    for k in config.__dict__:
        print(k, ': ',  config.__dict__[k])

    start_time = time.time()
    print(start_time)
    print("Loading data...")
    if 'cifar100' in args.model.lower():
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar100(config)
        train_iters = [build_cifar_iterator(train_data, config) for train_data in train_datas]
        dev_iters = [build_cifar_iterator(dev_data, config) for dev_data in dev_datas]
        test_iters = [build_cifar_iterator(test_data, config) for test_data in test_datas]
    elif 'cifar10' in args.model.lower():
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar10(config)
        train_iters = [build_cifar_iterator(train_data, config) for train_data in train_datas]
        dev_iters = [build_cifar_iterator(dev_data, config) for dev_data in dev_datas]
        test_iters = [build_cifar_iterator(test_data, config) for test_data in test_datas]
    else:
        vocab, train_datas, dev_datas, test_datas = build_dataset_from_csv(config, args.word)
        train_iters = [build_iterator(train_data, config) for train_data in train_datas]
        dev_iters = [build_iterator(dev_data, config) for dev_data in dev_datas]
        test_iters = [build_iterator(test_data, config) for test_data in test_datas]
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    if 'cifar' in model_name.lower():
        from torchvision import models
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, config.num_classes)
        model = model.to(config.device)
    else:
        model = x.Model(config).to(config.device)
        init_network(model)

    for i in range(config.task_number):
        print(datetime.datetime.now().strftime('%F %T'))
        if args.paradigm.lower() == 'cfed':
            train_CFeD_c(config, model, train_iters[i], dev_iters, copy.deepcopy(train_iters[-1]), i)
        elif args.paradigm == 'lwf':
            train_lwf(config, model, train_iters[i], dev_iters, copy.deepcopy(train_iters[-1]), i)
        elif args.paradigm == 'dmc':
            train_dmc(config, model, train_iters[i], dev_iters, copy.deepcopy(train_iters[-1]), i)
        elif args.paradigm == 'ewc':
            train_ewc(config, model, train_iters[i], dev_iters, i)
        elif args.paradigm == 'multihead' or model_name == 'TextCNN_multihead':
            train_multihead(config, model, train_iters[i], dev_iters, i)
        else:
            train(config, model, train_iters[i], dev_iters, i)
        for i in range(config.task_number):
            test(config, model, test_iters[i], i)