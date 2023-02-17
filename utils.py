from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torch.nn.functional as F
import copy
from collections import defaultdict
import numpy as np
import copy
import math
from torchvision import datasets, transforms
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 40, 1250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def get_dataset(dataset_name):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset_name == 'cifar10':
        data_dir = '../data/cifar/'
        train_transform =transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                                      transform=test_transform)

        user_groups = cifar_iid(train_dataset, 20)
        user_groups_test = cifar_iid(test_dataset, 20)

        return train_dataset, test_dataset, user_groups, user_groups_test

    if dataset_name == 'cinic-10':
        cinic_directory = '../data/cinic-10'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        train_dataset = datasets.ImageFolder(cinic_directory + '/train',
                                             transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                           transforms.RandomHorizontalFlip(),
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize(mean=cinic_mean,
                                                                                                std=cinic_std)]))
        test_dataset = datasets.ImageFolder(cinic_directory + '/test',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean=cinic_mean,
                                                                                              std=cinic_std)]))

        user_groups = cifar_iid(train_dataset, 20)
        user_groups_test = cifar_iid(test_dataset, 20)

        return train_dataset, test_dataset, user_groups, user_groups_test

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        #image = self.dataset[self.idxs[item]]
        return image, label

class DatasetSplitSpeech(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image = self.dataset[self.idxs[item]]
        return image

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def common_basic(w):

    minIndex = 0
    minLength = 10000000

    for i in range(0, len(w)):
        if len(w[i]) < minLength:
            minIndex = i
            minLength = len(w[i])

    commonList = [s for s in w[minIndex].keys()]

    for i in range(0, len(w)):
        local_weights_names = [s for s in w[i].keys()]
        for j in range(len(commonList)):
            if commonList[j] == local_weights_names[j]:
                continue
            else:
                del commonList[j:len(commonList)+1]
                break

    for k in commonList:
        comWeight = copy.deepcopy(w[0][k])
        for i in range(1, len(w)):
            comWeight += w[i][k]
        comWeight = comWeight / len(w)

        for i in range(0, len(w)):
            w[i][k] = comWeight

    return w, commonList

def common_max(w):

    w_copy = copy.deepcopy(w)

    count = [[] for i in range(len(w))]
    for i in range(len(w)):
        local_weights_names = [s for s in w[i].keys()]
        count[i] = [1 for m in range(len(local_weights_names))]

    for i in range(0, len(w)):

        local_weights_names1 = [s for s in w[i].keys()]

        for j in range(i+1, len(w)):
            if i == j:
                continue
            local_weights_names2 = [s for s in w[j].keys()]
            for k in range(0, len(local_weights_names1)):
                if local_weights_names2[k] == local_weights_names1[k]:
                    name = local_weights_names1[k]
                    w[i][name] += w_copy[j][name]
                    w[j][name] += w_copy[i][name]
                    count[i][k] += 1
                    count[j][k] += 1
                else:
                    break

    for c in range(0, len(w)):
        local_weights_names = [s for s in w[c].keys()]
        for k in range(0, len(local_weights_names)):
            w[c][local_weights_names[k]] = w[c][local_weights_names[k]].cpu() / count[c][k]

    return w

def local_train(net, dataset, idxs, device):

    net.to(device)
    net.train()

    # train and update
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    ldr_train = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=64, shuffle=True)
    for iter in range(10):
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)

            net.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return net.state_dict()

def test_inference(model, dataset, idxs, device):
    """ Returns the test accuracy and loss.
    """
    # model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    model.to(device)
    model.train()

    ldr_test = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=64, shuffle=False)

    for _, (images, labels) in enumerate(ldr_test):
        model.zero_grad()
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        # Prediction
        _, pred_labels = torch.max(outputs, 1)

        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct * 1.0 / total
    return accuracy