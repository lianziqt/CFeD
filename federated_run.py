# coding: UTF-8
import time
import torch
import torch.nn as nn
import numpy as np
from train_eval_fed import train, test, train_CFeD, train_ewc, train_multihead, train_lwf, train_DMC
from importlib import import_module
from utils import build_usergroup, get_parameter_number, init_network, init_network_resnet
import argparse
import copy

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextCNN_multihead')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--paradigm', default='', type=str, help='choose the training paradigm: CFeD, ewc, multihead. default is suqentially training')
parser.add_argument('--scenario', default='domain', type=str, help=':Class-IL or Domain-IL')
args = parser.parse_args()


if __name__ == '__main__':
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextCNN, TextCNN_multihead
    scenario = args.scenario.lower()
    from utils import build_dataset, build_iterator, get_time_dif, build_dataset_from_csv_fed, build_dataset_cifar10, \
        build_cifar_iterator, build_dataset_cifar100

    x = import_module('models.' + model_name)
    config = x.Config(scenario, embedding)
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
    if 'Cifar100' in args.model:
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar100(config)
    elif 'Cifar10' in args.model:
        vocab, train_datas, dev_datas, test_datas = build_dataset_cifar10(config)
    else:
        vocab, train_datas, dev_datas, test_datas = build_dataset_from_csv_fed(config, args.word)
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

    print(get_parameter_number(model))

    for i in range(config.task_number):
        if args.paradigm.lower() == 'cfed':
            train_CFeD(config, model, train_datas[i], dev_datas, i, copy.deepcopy(train_datas[-1]))
        elif args.paradigm.lower() == 'ewc':
            train_ewc(config, model, train_datas[i], dev_datas, i)
        elif args.paradigm.lower() == 'multihead' or model_name == 'TextCNN_multihead':
            train_multihead(config, model, train_datas[i], dev_datas, i)
        elif args.paradigm.lower() == 'dmc':
            train_DMC2(config, model, train_datas[i], dev_datas, i, copy.deepcopy(train_datas[-1]))
        elif args.paradigm.lower() == 'lwf':
            train_lwf(config, model, train_datas[i], dev_datas, i)
        else:
            train(config, model, train_datas[i], dev_datas, i)
        for i in range(config.task_number):
            test(config, model, test_datas[i], i)


    # for k in [2, 5, 10, 15, 20]:
    #     config.surrogate_ratio = k
    #     if 'cifar'  in model_name.lower():
    #         from torchvision import models
    #         model = models.resnet18(pretrained=True)
    #         model.fc = torch.nn.Linear(512, config.num_classes)
    #         model = model.to(config.device)
    #     else:
    #         model = x.Model(config).to(config.device)
    #         init_network(model)
    #     for i in range(config.task_number):
    #         train_CFeD(config, model, train_datas[i], dev_datas, i, copy.deepcopy(train_datas[-1]))
    #         acc_list = []
    #         for j in range(config.task_number):
    #             acc_list.append(test(config, model, test_datas[j], j))
                
    #         print("Surrogate_Ratio:{0}, New Task ({1}), Acc: {2:>6.2%}".format(k, i, acc_list[i]))
    #         print("Surrogate_Ratio:{0}, Avg Task ({1}) Acc: {2:>6.2%}".format(k, i, sum(acc_list[:(i+1)]) / (i + 1.0)))
