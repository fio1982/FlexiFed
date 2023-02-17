#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import torch

from utils.utils import *
from models.vgg import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    device = torch.device("cuda:0")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, idx_test = get_dataset("cifar10")

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    number_device = 8
    idxs_users = [_id for _id in range(number_device)]
    modelAccept = {_id: None for _id in range(number_device)}

    for _id in range(number_device):
        if _id < 2:
            modelAccept[_id] = vgg11_bn()

        elif _id >= 2 and _id < 4:
            modelAccept[_id] = vgg13_bn()

        elif _id >= 4 and _id < 6:
            modelAccept[_id] = vgg16_bn()

        else:
            modelAccept[_id] = vgg19_bn()

    localData_length = len(user_groups[0]) / 10
    start = 0

    local_acc = [[] for i in range(number_device)]
    for epoch in range(502):

        print(f'\n | Global Training Round : {epoch+1} |\n')
        end = start + localData_length

        for idx in idxs_users:

            idx_train_all = list(user_groups[idx])
            idx_train_batch = set(idx_train_all[int(start):int(end)])

            if epoch == 0:
                model = modelAccept[idx]

            if epoch > 0:
                if idx < 2:
                    model = vgg11_bn()
                    # model.load_state_dict(A)
                elif idx >= 2 and idx < 4:
                    model = vgg13_bn()
                    # model.load_state_dict(B)

                elif idx >= 4 and idx < 6:
                    model = vgg16_bn()
                    # model.load_state_dict(C)

                else:
                    model = vgg19_bn()
                    # model.load_state_dict(D)

                model.load_state_dict(modelAccept[idx])

            acc = test_inference(model, test_dataset, list(idx_test[idx]), device)
            local_acc[idx].append(round(acc, 2))
            if epoch % 10 == 0:
                print(local_acc[idx])

            Model = copy.deepcopy(model)
            localModel = local_train(Model, train_dataset, idx_train_batch, device)
            modelAccept[idx] = copy.deepcopy(localModel)

        start = end % 2500

        modelAccept = common_max(modelAccept)
        # modelAccept, _ = common_basic(modelAccept)
