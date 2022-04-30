#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from utils import build_dataset, build_iterator, get_time_dif, build_dataset_from_csv_fed, init_network
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from utils import build_usergroup, build_usergroup_non_iid
from update import LocalUpdate, test_inference, GlobalUpdate
from utils import average_weights, exp_details


def train(config, global_model, train_dataset, dev_datasets, current_task, combin=False):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    exp_details(config)

    device = config.device

    # load dataset and user groups
    if config.iid:
        user_groups = build_usergroup(train_dataset, config)
    else:
        user_groups = build_usergroup_non_iid(train_dataset, config)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    for epoch in tqdm(range(config.num_epochs)):
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[idx], logger=logger,
                                      current_task=current_task)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[c], logger=logger,
                                      current_task=current_task)
            acc, loss = local_model.inference(model=global_model)

            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for i in range(config.task_number):
            # Test inference after completion of training
            test_acc, test_loss = test_inference(config, global_model, dev_datasets[i], i)
            print(' \n Results after {} global rounds of training:'.format(config.num_epochs))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            writer.add_scalar("acc/dev{}".format(i), 100*test_acc, epoch)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def train_lwf(config, global_model, train_dataset, dev_datasets, current_task, combin=False):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    exp_details(config)

    device = config.device

    # load dataset and user groups
    if config.iid:
        user_groups = build_usergroup(train_dataset, config)
    else:
        user_groups = build_usergroup_non_iid(train_dataset, config)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    initial_model = copy.deepcopy(global_model)
    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    for epoch in tqdm(range(config.num_epochs)):
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[idx], logger=logger,
                                      current_task=current_task)
            w, loss = local_model.update_weights_lwf(
                model=copy.deepcopy(global_model), initial_model=initial_model, global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[c], logger=logger,
                                      current_task=current_task)
            acc, loss = local_model.inference(model=global_model)

            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for i in range(config.task_number):
            # Test inference after completion of training
            test_acc, test_loss = test_inference(config, global_model, dev_datasets[i], i)
            print(' \n Results after {} global rounds of training:'.format(config.num_epochs))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            writer.add_scalar("acc/dev{}".format(i), 100*test_acc, epoch)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


def train_CFeD(config, global_model, train_dataset, dev_datasets, current_task, surrogate_data):
    start_time = time.time()
    logger = SummaryWriter('../logs')
    exp_details(config)
    device = config.device

    # load dataset and user groups
    if config.iid:
        user_groups = build_usergroup(train_dataset, config)
    else:
        user_groups = build_usergroup_non_iid(train_dataset, config)
    surrogate_groups = build_usergroup(surrogate_data, config)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]_client'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    # random choose a shard for server distillation
    global_idx = np.random.choice(range(config.num_users), 1, replace=False)
    init_model = copy.deepcopy(global_model)

    for epoch in tqdm(range(config.num_epochs)):
        local_weights, local_losses, local_models = [], [], []

        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        if current_task == 0:
            idxs_users = np.random.choice(range(config.num_users), m, replace=False)
        else:
            idxs_users = np.random.choice(range(config.num_users), int(m * 0.6), replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[idx], logger=logger,
                                      current_task=current_task)
            w, loss, model = local_model.update_weights_CFeD_new(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_models.append(copy.deepcopy(model))

        if current_task > 0:
            idxs_users = np.random.choice(range(config.num_users), int(m*0.4), replace=False)
            for idx in idxs_users:
                local_model = LocalUpdate(args=config, train_data=surrogate_data,
                                          idxs=surrogate_groups[idx], logger=logger,
                                          current_task=current_task)
                w, loss, model = local_model.update_weights_CFeD_old(
                    model=copy.deepcopy(global_model), global_round=epoch, initial_model=init_model)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                local_models.append(copy.deepcopy(model))

        if config.server_distillation:
            local_models.append(copy.deepcopy(global_model))
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        if config.server_distillation:

            local_model = GlobalUpdate(args=config, train_data=surrogate_data,
                                      idxs=surrogate_groups[global_idx[0]], logger=logger,
                                      current_task=current_task, local_models=local_models)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            global_model.load_state_dict(w)


        # return
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[c], logger=logger,
                                      current_task=current_task)
            acc, loss = local_model.inference(model=global_model)

            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for i in range(config.task_number):
            # Test inference after completion of training
            test_acc, test_loss = test_inference(config, global_model, dev_datasets[i], i)
            print(' \n Results after {} global rounds of training:'.format(config.num_epochs))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            writer.add_scalar("acc/dev{}".format(i), 100*test_acc, epoch)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))



def train_ewc(config, global_model, train_dataset, dev_datasets, current_task):
    start_time = time.time()
    logger = SummaryWriter('../logs')
    exp_details(config)

    device = config.device

    # load dataset and user groups
    if config.iid:
        user_groups = build_usergroup(train_dataset, config)
    else:
        user_groups = build_usergroup_non_iid(train_dataset, config)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]_ewc'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    for epoch in tqdm(range(config.num_epochs)):
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[idx], logger=logger,
                                      current_task=current_task)
            w, loss = local_model.update_weights_ewc(
                model=copy.deepcopy(global_model), global_round=epoch)

            w_temp = copy.deepcopy(w)
            for k in w.keys():
                if '__' in k:
                    w_temp.pop(k)
            local_weights.append(w_temp)
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[c], logger=logger,
                                      current_task=current_task)
            acc, loss = local_model.inference(model=global_model)

            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for i in range(config.task_number):
            # Test inference after completion of training
            test_acc, test_loss = test_inference(config, global_model, dev_datasets[i], i)
            print(' \n Results after {} global rounds of training:'.format(config.num_epochs))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            writer.add_scalar("acc/dev{}".format(i), 100*test_acc, epoch)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


def train_multihead(config, global_model, train_dataset, dev_datasets, current_task):
    start_time = time.time()
    logger = SummaryWriter('../logs')

    exp_details(config)

    device = config.device

    # load dataset and user groups
    if config.iid:
        user_groups = build_usergroup(train_dataset, config)
    else:
        user_groups = build_usergroup_non_iid(train_dataset, config)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    for epoch in tqdm(range(config.num_epochs)):
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[idx], logger=logger,
                                      current_task=current_task)
            w, loss = local_model.update_weights_multihead(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[c], logger=logger,
                                      current_task=current_task)
            acc, loss = local_model.inference(model=global_model)

            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for i in range(config.task_number):
            # Test inference after completion of training
            test_acc, test_loss = test_inference(config, global_model, dev_datasets[i], i)
            print(' \n Results after {} global rounds of training:'.format(config.num_epochs))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            writer.add_scalar("acc/dev{}".format(i), 100*test_acc, epoch)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

def train_DMC(config, global_model, train_dataset, dev_datasets, current_task, surrogate_data):
    start_time = time.time()
    logger = SummaryWriter('../logs')
    exp_details(config)
    device = config.device

    # load dataset and user groups
    if config.iid:
        user_groups = build_usergroup(train_dataset, config)
    else:
        user_groups = build_usergroup_non_iid(train_dataset, config)
    surrogate_groups = build_usergroup(surrogate_data, config)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    
    print(global_model)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    writer = SummaryWriter(log_dir=config.log_path + '/' +
                                   time.strftime('%m-%d_%H.%M_{}_{}_C[{}]_E[{}]_B[{}]_client'
                                                 .format(config.model_name, config.num_epochs, config.frac,
                                                         config.local_ep, config.local_bs),
                                                 time.localtime()))
    # random choose a shard for server distillation
    global_idx = np.random.choice(range(config.num_users), 1, replace=False)
    init_model = copy.deepcopy(global_model)

    for epoch in tqdm(range(config.num_epochs)):
        local_weights, local_losses, local_models = [], [], []

        print('\n | Global Training Round : {}|\n'.format(epoch+1))

        global_model.train()
        local_model_on_new_task = copy.deepcopy(global_model)
        init_network(local_model_on_new_task)
        local_model_on_new_task.train()
        m = max(int(config.frac * config.num_users), 1)
        idxs_users = np.random.choice(range(config.num_users), m, replace=False)


        for idx in idxs_users:
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[idx], logger=logger,
                                      current_task=current_task)
            w, loss, model = local_model.update_weights_DMC_new(
                model=copy.deepcopy(local_model_on_new_task), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_models.append(copy.deepcopy(model))
        
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        if current_task > 0:
        #     idxs_users = np.random.choice(range(config.num_users), int(m*0.4), replace=False)
            for i, idx in enumerate(idxs_users):
                local_model = LocalUpdate(args=config, train_data=surrogate_data,
                                          idxs=surrogate_groups[idx], logger=logger,
                                          current_task=current_task)
                # print(len(local_models))
                # print(idx)
                w, loss, model = local_model.update_weights_DMC_combine(
                    global_model=copy.deepcopy(init_model), new_local_model=copy.deepcopy(global_model), global_round=epoch)
                local_weights[i] = copy.deepcopy(w)
                local_losses[i] = copy.deepcopy(loss)
                local_models[i] = copy.deepcopy(model)

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # return
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.num_users):
            local_model = LocalUpdate(args=config, train_data=train_dataset,
                                      idxs=user_groups[c], logger=logger,
                                      current_task=current_task)
            acc, loss = local_model.inference(model=global_model)

            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("loss/train_{}".format(current_task), np.mean(np.array(train_loss)), epoch)
            writer.add_scalar("acc/train_{}".format(current_task), 100*train_accuracy[-1], epoch)

        for i in range(config.task_number):
            # Test inference after completion of training
            test_acc, test_loss = test_inference(config, global_model, dev_datasets[i], i)
            print(' \n Results after {} global rounds of training:'.format(config.num_epochs))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            writer.add_scalar("acc/dev{}".format(i), 100*test_acc, epoch)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


def test(config, model, test_iter, current_task):
    # test
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter, current_task, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, current_task, test=False):
    acc, loss_total = test_inference(config, model, data_iter, current_task)
    return acc, loss_total / len(data_iter)