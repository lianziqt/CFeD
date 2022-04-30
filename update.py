#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from loss import DistillationLoss
from elastic_weight_consolidation import ElasticWeightConsolidation
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # # for image
        # return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        # # for text
        return torch.tensor(image), torch.tensor(label)


class MyDataSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        # # for image
        # return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        # # for text
        return torch.tensor(image), torch.tensor(label)

class GlobalDataSetSplit(Dataset):
    def __init__(self, dataset, idxs, local_models, device):
        self.device = device
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.local_models = local_models
        for i in self.local_models:
            i.eval()
        self.model_idx = 0

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        temp_image, label = torch.tensor(image).to(self.device), torch.tensor(label).to(self.device)
        temp_image = temp_image.view(1, temp_image.shape[0])
        soft_label = self.local_models[self.model_idx](temp_image)
        self.model_idx = (self.model_idx + 1) % len(self.local_models)
        return torch.tensor(image), soft_label.view(soft_label.shape[1])


class LocalUpdate(object):
    def __init__(self, args, train_data, idxs, logger, current_task):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            train_data, list(idxs))
        self.device = args.device
        self.current_task = current_task
        self.criterion = F.cross_entropy

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=False, drop_last=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/self.args.local_bs), shuffle=False, drop_last=True)
        # idxs_test -> idxs_val
        testloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                batch_size=int(len(idxs_val)/self.args.local_bs), shuffle=False, drop_last=True)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_prob = model(images)
                loss = self.criterion(log_prob, labels)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_multihead(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                log_prob = log_probs[self.current_task]
                loss = self.criterion(log_prob, labels)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_CFeD_new(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        model.train()
        epoch = self.args.local_ep
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_prob = model(images)
                loss = self.criterion(log_prob, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model

    def update_weights_CFeD_old(self, model, initial_model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        dis_loss = DistillationLoss()
        old_targets = []
        model.eval()
        with torch.no_grad():
            for trains, labels in self.trainloader:
                trains, labels = trains.to(self.device), labels.to(self.device)
                output = initial_model(trains)
                old_targets.append(output)

        model.train()
        for iter in range(int(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # labels -= int(self.args.task_list[self.current_task][0])
                model.zero_grad()
                output = model(images)
                loss = dis_loss(output, old_targets[batch_idx], 2.0, 0.1)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model

    def update_weights_lwf(self, model, initial_model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        dis_loss = DistillationLoss()
        old_targets = []
        model.eval()
        if self.current_task > 0:
            with torch.no_grad():
                for trains, labels in self.trainloader:
                    trains, labels = trains.to(self.device), labels.to(self.device)
                    output = initial_model(trains)
                    old_targets.append(output)

        model.train()
        for iter in range(int(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # labels -= int(self.args.task_list[self.current_task][0])
                model.zero_grad()
                output = model(images)
                loss = self.criterion(output, labels)
                if self.current_task > 0:
                    loss += dis_loss(output, old_targets[batch_idx], 2.0, 0.1)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model

    def update_weights_ewc(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        ewc = ElasticWeightConsolidation(model, weight=100000)
        if self.current_task > 0:
            ewc.register_ewc_params(self.trainloader, self.args.local_bs, len(self.trainloader))
            
        model.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                if self.current_task > 0:
                    loss += ewc.consolidation_loss(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                batch_num = batch_idx
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
    def update_weights_DMC_new(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        model.train()
        epoch = self.args.local_ep
        # if self.args.server_distillation and self.current_task > 0:
        #     epoch *= 3
        for iter in range(epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_prob = model(images)
                loss = self.criterion(log_prob, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model

    def update_weights_DMC_combine(self, global_model, new_local_model, global_round):
        # Set mode to train model
        # model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        # optimizer = torch.optim.Adam(global_model.parameters(), lr=self.args.learning_rate * 10)
        # dmc_loss = DMCLoss(self.current_task)
        dmc_loss = nn.MSELoss()
        old_targets, new_targets = [], []
        global_model.eval()
        new_local_model.eval()
        with torch.no_grad():
            for trains, labels in self.trainloader:
                trains, labels = trains.to(self.device), labels.to(self.device)
                old_targets.append(global_model(trains))
                new_targets.append(new_local_model(trains))

        new_global_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(new_global_model.parameters(), lr=self.args.learning_rate * 10)
        new_global_model.train()
        
        init_network(new_global_model)
        for iter in range(self.args.local_ep * 2):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                new_global_model.zero_grad()
                log_prob = new_global_model(images)

                outputs_old = old_targets[batch_idx]
                outputs_cur = new_targets[batch_idx]
                outputs_old -= outputs_old.mean(dim=1).reshape(images.shape[0],-1)
                outputs_cur -= outputs_cur.mean(dim=1).reshape(images.shape[0],-1)
                # outputs_tot = torch.cat((outputs_old, outputs_cur), dim=1)
                outputs_tot = outputs_old + outputs_cur
                loss = dmc_loss(log_prob, outputs_tot)
                # loss = dmc_loss(log_prob, old_targets[batch_idx], new_targets[batch_idx])
                # print(loss)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return new_global_model.state_dict(), sum(epoch_loss) / len(epoch_loss), new_global_model

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # Inference
            log_prob = model(images)
            if isinstance(log_prob, list):
                log_prob = log_prob[self.current_task]
                labels -= int(self.args.task_list[self.current_task][0])

            batch_loss = self.criterion(log_prob, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(log_prob, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        print(accuracy)
        return accuracy, loss


class GlobalUpdate(object):
    def __init__(self, args, train_data, idxs, logger, current_task, local_models):
        self.args = args
        self.logger = logger
        self.local_models = local_models
        self.device = args.device
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            train_data, list(idxs))
        self.current_task = current_task
        self.criterion = F.cross_entropy

    def train_val_test(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):]

        trainloader = DataLoader(GlobalDataSetSplit(dataset, idxs_train, local_models=self.local_models, device=self.device),
                                 batch_size=128, shuffle=False)
        validloader = DataLoader(GlobalDataSetSplit(dataset, idxs_val, local_models=self.local_models, device=self.device),
                                 batch_size=int(len(idxs_val) / 10), shuffle=False)
        # idxs_test -> idxs_val
        testloader = DataLoader(GlobalDataSetSplit(dataset, idxs_val, local_models=self.local_models, device=self.device),
                                batch_size=int(len(idxs_val) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)

        model.train()
        dis_loss = DistillationLoss()
        for local_model in self.local_models:
            local_model.eval()
        epoch = self.args.local_ep
        for iter in range(int(epoch)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                outputs = model(images)
                loss = dis_loss(outputs, labels, 2.0, 0.1)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(args, model, test_dataset, current_task):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = F.cross_entropy
    testloader = DataLoader(MyDataSet(test_dataset), batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        outputs = model(images)

        if isinstance(outputs, list):
            outputs = outputs[current_task]
            labels -= int(args.task_list[current_task][0])
        batch_loss = criterion(outputs, labels)

        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


