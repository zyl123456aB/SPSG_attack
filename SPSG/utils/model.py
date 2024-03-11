#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import random
import time
from datetime import datetime
from collections import defaultdict as dd
import pickle
import numpy as np
import math
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models
import SPSG.config as cfg
import SPSG.utils.utils as SPSG_utils



def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def train_step(model,blackbox, train_loader, criterion, optimizer, epoch, device, log_interval=10, writer=None):
    model.train()
    train_loss = 0.
    train_losskl = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    # t_start = time.time()
    nans = 0
    L1loss = nn.L1Loss()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        logits = []
        sgs = []
        labels = []
        for idx in  range(inputs.shape[0]):
            with open(targets[idx], 'rb') as rf:
                real_targets = pickle.load(rf)
                logits.append(real_targets[0])
                labels.append(real_targets[1])
                sgs.append(real_targets[2])
        logits = torch.stack(logits)
        sgs = torch.stack(sgs)
        labels = torch.stack(labels)
        inputs, sgs = inputs.to(device),sgs.to(device)
        sgs = sgs.view(sgs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
        logits = logits.view(sgs.shape[0], -1)
        if torch.isnan(sgs).any() or torch.isnan(logits).any():
            sgs = sgs.reshape(sgs.shape[0],-1)
            logits = logits.reshape(sgs.shape[0],-1)
            mask = ~(torch.any(sgs.isnan(),dim=1)|torch.any(logits.isnan(),dim=1))
            sgs = sgs[mask]
            sgs =  sgs.view(sgs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3])
            logits = logits[mask].view(sgs.shape[0],-1)
            inputs =  inputs[mask]
            nans = nans+1
            print(sgs.shape)


        optimizer.zero_grad()
        inputs.requires_grad_()
        outputs = model(inputs)
        _, predict = outputs.max(1)
        with torch.no_grad():
            box_outputs = blackbox(inputs).clone().detach()
            _, box_predict = box_outputs.max(1)

        lossz = F.cross_entropy(outputs,box_predict)
        newreg = torch.autograd.grad(lossz, inputs, create_graph=True)[0].view(inputs.shape)

        inputs_p = []
        Rsamples =[]
        types = 10000
        sgs = calculate(sgs).view(inputs.shape)
        for j in range(sgs.shape[0]):
            a = sgs[j].unique()
            if types >= a.shape[0]:
                types = a.shape[0]
            Rsamples.append(random.sample(list(range(a.shape[0])), a.shape[0]))
        for time in range(types):
            inputs_p.append(inputs.clone().detach())

        losskk = []
        for xiao,input_p in enumerate(inputs_p):
            for j in range(sgs.shape[0]):
                a = sgs[j].unique()
                input_p[j,sgs[j]==a[Rsamples[j][xiao]]] = input_p[j,sgs[j]==a[Rsamples[j][xiao]]] + 1e-4
            outputs_p = model(input_p)
            with torch.no_grad():
                box_outputs_p = blackbox(input_p).clone().detach()
                _, predict_p = box_outputs_p.max(1)
            if criterion==None:
                losskk.append(F.cross_entropy(outputs_p, predict_p) + 0.1 * L1loss(
                    torch.index_select(F.softmax(outputs_p, dim=1), 1, predict_p),
                    torch.index_select(box_outputs_p, 1, predict_p)) )
            else:
                losskk.append(F.cross_entropy(outputs_p, predict_p) + 0.1 * criterion(
                    F.softmax(outputs, dim=1), box_outputs))



        for j in range(sgs.shape[0]):
            a = sgs[j].unique()

            if a.shape[0] > 49 * 3:
                print("a", a.shape)
                print("sgs[j]", torch.isnan(sgs[j]).any())
            for i in range(a.shape[0]):
                newreg[j, sgs[j] == a[i]] = newreg[j, sgs[j] == a[i]].view(-1).mean(dim=0)


        newreg = calculate2(newreg,sgs)
        newreg = newreg.view(sgs.shape[0],-1)
        sgs = sgs.view(sgs.shape[0],-1)
        losskl = 1 - (F.cosine_similarity(newreg,
                                          sgs, dim=1).sum()) / (sgs.shape[0])
        if criterion==None:
            loss1 = F.cross_entropy(outputs, box_predict)+0.1*L1loss( torch.index_select(F.softmax(outputs, dim=1),1,box_predict),torch.index_select(F.softmax(box_outputs, dim=1),1,box_predict))
        else:
            loss1 = F.cross_entropy(outputs, box_predict) + 0.1 * criterion(
                F.softmax(outputs, dim=1),box_outputs)

        a = math.exp(loss1.item()-losskl.item()-3)

        if a<0.1: a = 0.1
        loss  = loss1 + sum(losskk) + a*losskl
        loss.backward()
        optimizer.step()









        if writer is not None:
            pass

        train_loss += loss1.item()
        train_losskl += losskl.item()
        _, predicted = outputs.max(1)
        total += logits.size(0)

        correct += predicted.eq(box_predict).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total

        train_loss_batch = train_loss / total
        train_losskl_batch = train_losskl / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Losskl: {:.6f}\tAccuracy: {:.1f} ({}/{}) a:{}'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss1.item(),losskl.item(), acc, correct, total,a))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    # t_end = time.time()
    # t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc


def test_step(model, blackbox, test_loader, criterion, device, epoch=0., silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    correct2 = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            agreementputs = blackbox(inputs)
            loss = criterion(outputs, targets)
            nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, targets2 = agreementputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            correct2 += predicted.eq(targets2).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    acc2 = 100. * correct2 / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{}) \tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
                                                                             correct, total,acc2,
                                                                             correct2, total))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc

def calculate(image):
    # The SGP module

    N, C, H, W = image.shape[:4]
    image = image.view(N, C, -1)
    image[image == 0] = 1e-14

    maxpool,_ = image.max(2)
    maxpool = maxpool.unsqueeze(2).expand(N,C,H*W)/2.0
    minpool,_ = image.min(2)
    minpool = minpool.unsqueeze(2).expand(N,C,H*W)/2.0
    image = torch.where((image>=maxpool)|(image<=minpool),image,0)
    image = torch.where((image>=maxpool), image / (2*maxpool),image)
    image = torch.where((image<=minpool), image / (2*maxpool), image)
    return image

def calculate2(image,sgs):

    image2 = image.clone().detach()*(sgs!=0)
    N, C, H, W = image.shape[:4]
    image = image.view(N, C, -1)
    image2 = image2.view(N, C, -1)
    maxpool,_ = image2.abs().max(2)
    maxpool = maxpool.unsqueeze(2).expand(N,C,H*W)

    image =  image / (maxpool)
    return image


def train_model(model, blackbox,trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        SPSG_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step(model,blackbox, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval)
        scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None:
            test_loss, test_acc = test_step(model,blackbox, test_loader, criterion_test, device, epoch=epoch)
            best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model
