#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, init_network
from tensorboardX import SummaryWriter
from loss import DistillationLoss
from elastic_weight_consolidation import  ElasticWeightConsolidation


def train(config, model, train_iter, dev_iters, current_task):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + config.train_tasks[current_task] + '/'
                                   + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for index, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # print eval information
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                writer.add_scalar("loss/train_{}".format(current_task), loss.item(), total_batch)
                writer.add_scalar("acc/train_{}".format(current_task), train_acc, total_batch)
                for i in range(config.task_number):
                    dev_acc, dev_loss = evaluate(config, model, dev_iters[i], i)
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  ' \
                          'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                    writer.add_scalar("loss/dev_{}".format(i), dev_loss, total_batch)
                    writer.add_scalar("acc/dev_{}".format(i), dev_acc, total_batch)
                model.train()
            total_batch += 1
    writer.close()


def train_ewc(config, model, train_iter, dev_iters, current_task):
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    ewc = ElasticWeightConsolidation(model, weight=100000)
    total_batch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if current_task > 0:
        ewc.register_ewc_params(train_iter, config.batch_size, config.num_epochs)
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for index, (trains, labels) in enumerate(train_iter):
            output = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(output, labels)
            if current_task > 0:
                loss += ewc.consolidation_loss(output, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # print eval information
                true = labels.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                writer.add_scalar("loss/train_{}".format(current_task), loss.item(), total_batch)
                writer.add_scalar("acc/train_{}".format(current_task), train_acc, total_batch)
                for i in range(config.task_number):
                    dev_acc, dev_loss = evaluate(config, model, dev_iters[i], i)
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                    writer.add_scalar("loss/dev_{}".format(i), dev_loss, total_batch)
                    writer.add_scalar("acc/dev{}".format(i), dev_acc, total_batch)
                model.train()
            total_batch += 1
    
    writer.close()


def train_multihead(config, model, train_iter, dev_iters, current_task):
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for index, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            labels.to(config.device)
            labels -= int(config.task_list[current_task][0])
            loss = F.cross_entropy(outputs[current_task], labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # print eval information
                true = labels.data.cpu()
                predic = torch.max(outputs[current_task].data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)

                writer.add_scalar("loss/train_{}".format(current_task), loss.item(), total_batch)
                writer.add_scalar("acc/train_{}".format(current_task), train_acc, total_batch)
                for i in range(config.task_number):
                    dev_acc, dev_loss = evaluate(config, model, dev_iters[i], i)
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                    writer.add_scalar("loss/dev_{}".format(i), dev_loss, total_batch)
                    writer.add_scalar("acc/dev{}".format(i), dev_acc, total_batch)
                model.train()
            total_batch += 1
    writer.close()


def train_CFeD_c(config, model, train_iter, dev_iters, surrogate_iter, current_task):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    # collect the old outputs
    lens_train = len(train_iter)
    dis_loss = DistillationLoss()
    old_targets = []
    model.eval()
    print('Epoch old output')
    if current_task > 0:
        i = 0
        with torch.no_grad():
            for trains, labels in surrogate_iter:
                output = model(trains)
                old_targets.append((trains, output, True))
                i += 1
                if i == lens_train: # skip the final incomplete batch
                    break

    mix_data = [(trains, labels, False) for index, (trains, labels) in enumerate(train_iter)]
    if current_task > 0:
        mix_data.extend(old_targets)
    np.random.shuffle(mix_data)
    print('Epoch old output')
    model.train()
    epoch = config.num_epochs
    for epoch in range(epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        print(len(mix_data))
        for index, (trains, labels, is_distillation) in enumerate(mix_data):
            output = model(trains)
            model.zero_grad()
            # if isinstance(output, list):
            #     labels -= int(config.task_list[current_task][0])
            labels.to(config.device)
            if is_distillation:
                loss = dis_loss(output, labels, 2.0, 0.1)
            else:
                loss = F.cross_entropy(output, labels)

            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0 and not is_distillation:
                # print eval information
                true = labels.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                writer.add_scalar("loss/train_{}".format(current_task), loss.item(), total_batch)
                writer.add_scalar("acc/train_{}".format(current_task), train_acc, total_batch)
                for i in range(config.task_number):
                    dev_acc, dev_loss = evaluate(config, model, dev_iters[i], i)
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  common_Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                    writer.add_scalar("loss/dev_{}".format(i), dev_loss, total_batch)
                    writer.add_scalar("acc/dev{}".format(i), dev_acc, total_batch)
                model.train()
            total_batch += 1
    writer.close()

def train_lwf(config, model, train_iter, dev_iters, current_task):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    # collect the old outputs
    lens_train = len(train_iter)
    dis_loss = DistillationLoss()
    old_targets = []
    model.eval()
    print('Epoch old output')
    if current_task > 0:
        i = 0
        with torch.no_grad():
            for trains, labels in train_iter:
                output = model(trains)
                # outputs:（tasknumber, batchsize, single_out_dim）
                old_targets.append((trains, output, True))
                i += 1
                if i == lens_train: # skip the final incomplete batch
                    break

    print('Epoch old output')
    model.train()
    epoch = config.num_epochs
    for epoch in range(epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for index, (trains, labels) in enumerate(train_iter):
            output = model(trains)
            model.zero_grad()
            labels.to(config.device)
            loss = F.cross_entropy(output, labels)
            loss += dis_loss(output, old_targets[index], 2.0, 0.1)

            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                # print eval information
                true = labels.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                writer.add_scalar("loss/train_{}".format(current_task), loss.item(), total_batch)
                writer.add_scalar("acc/train_{}".format(current_task), train_acc, total_batch)
                for i in range(config.task_number):
                    dev_acc, dev_loss = evaluate(config, model, dev_iters[i], i)
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  common_Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                    writer.add_scalar("loss/dev_{}".format(i), dev_loss, total_batch)
                    writer.add_scalar("acc/dev{}".format(i), dev_acc, total_batch)
                model.train()
            total_batch += 1
    writer.close()

def train_dmc(config, model, train_iter, dev_iters, surrogate_iter, current_task):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.schedule_gamma)

    total_batch = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    # collect the old outputs
    lens_train = len(train_iter)
    dis_loss = torch.nn.MSELoss()
    old_targets = []
    model.eval()
    print('Epoch old output')
    if current_task > 0:
        with torch.no_grad():
            for trains, labels in surrogate_iter:
                output = model(trains)
                # outputs:（tasknumber, batchsize, single_out_dim）
                old_targets.append(output)


    print('Epoch old output')
    new_model = copy.deepcopy(model)
    new_model.train()
    epochs = config.num_epochs
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=config.learning_rate)
    # new model learn the new task
    for epoch in range(epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # print(len(mix_data))
        for index, (trains, labels) in enumerate(train_iter):
            output = new_model(trains)
            new_model.zero_grad()
            labels.to(config.device)
            # print(labels)
            loss = F.cross_entropy(output, labels)
            # print(loss)
            loss.backward()
            new_optimizer.step()

            if total_batch % 100 == 0:
                # print eval information
                true = labels.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                writer.add_scalar("loss/train_{}".format(current_task), loss.item(), total_batch)
                writer.add_scalar("acc/train_{}".format(current_task), train_acc, total_batch)
                for i in range(config.task_number):
                    dev_acc, dev_loss = evaluate(config, new_model, dev_iters[i], i)
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  common_Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                    writer.add_scalar("loss/dev_{}".format(i), dev_loss, total_batch)
                    writer.add_scalar("acc/dev{}".format(i), dev_acc, total_batch)
                new_model.train()
            total_batch += 1

    new_targets = []
    print(current_task)
    if current_task == 0:
        print('ok')
        model = new_model
        writer.close()
        return
    print('combine')
    with torch.no_grad():
        for trains, labels in surrogate_iter:
            new_output = new_model(trains)
            # outputs:（tasknumber, batchsize, single_out_dim）
            new_targets.append(new_output)       
    model.train()
    for epoch in range(epochs):
        print('DMC combine Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # print(len(mix_data))
        for index, (trains, labels) in enumerate(surrogate_iter):
            output = model(trains)
            model.zero_grad()
            labels.to(config.device)
            outputs_old = old_targets[index]
            # print("old ", outputs_old)
            outputs_cur = new_targets[index]
            # print("cur ", outputs_cur)
            outputs_old -= outputs_old.mean(dim=1).reshape(trains.shape[0],-1)
            outputs_cur -= outputs_cur.mean(dim=1).reshape(trains.shape[0],-1)
            outputs_tot = outputs_old + outputs_cur
            loss = dis_loss(output, outputs_tot)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                # print eval information
                true = labels.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                writer.add_scalar("loss/train_{}".format(current_task), loss.item(), total_batch)
                writer.add_scalar("acc/train_{}".format(current_task), train_acc, total_batch)
                for i in range(config.task_number):
                    dev_acc, dev_loss = evaluate(config, model, dev_iters[i], i)
                    time_dif = get_time_dif(start_time)
                    msg = 'DMC combine Iter: {0:>6},  common_Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif))
                    writer.add_scalar("loss/dev_{}".format(i), dev_loss, total_batch)
                    writer.add_scalar("acc/dev{}".format(i), dev_acc, total_batch)
                model.train()
            total_batch += 1
        # scheduler.step()
    writer.close()

def test(config, model, test_iter, current_task):
    # test based on the evaluate function
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, current_task, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, current_task, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            if isinstance(outputs, list):
                labels -= int(config.task_list[current_task][0])
                outputs = outputs[current_task]
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4,
                                               labels=list(range(config.num_classes)))
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)