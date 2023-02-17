'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
#from torchsummary import summary

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 10),
        )
        #fc: 1024, 4096, 512, 96
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))


# def test():
#     # net = VGG_11()
#
#     # net = VGG_16()
#     net = vgg11_bn()
#     names = [s for s in net.state_dict().keys() if s.startswith('classifier')]
#
#     # summary(net, (3, 32, 32))
#     # for name in net.state_dict():
#     #     print(name, '\t', net.state_dict()[name].size())
#
# test()
#

# from common.utils import *
# net1 = vgg13_bn().state_dict()
# net2 = vgg16_bn().state_dict()
# #
# # print(len(net1))
# # print(len(net2))
# #
#
# modelAccept = {_id: None for _id in range(2)}
# names1 = [s for s in net1.keys() if s.startswith('feature')]
# names2 = [s for s in net2.keys() if s.startswith('feature')]
#
# print(names2)
# del names2[14:len(names2)]
# print(names2)

# modelAccept[0] = net1
# modelAccept[1] = net2
#
# _, list = commonFedAvg_all_same(modelAccept)
# print(list)
# #
# # l = []
# # l.append(net1)
# # l.append(net2)
# # print(min(l, key=len))
#
#

# net1 = vgg16_bn().state_dict()
# net2 = vgg19_bn().state_dict()
# local_weights_names1 = [s for s in net1.keys()]
#
# local_weights_names2 = [s for s in net2.keys() if s.startswith('classifier')]
#
# print(local_weights_names1)
# print(local_weights_names2)
#
#
# common_list = []
#
# for i in range(len(local_weights_names1)):
#     if local_weights_names1[i] == local_weights_names2[i]:
#         common_list.append(local_weights_names2[i])
#     else:
#         break;
#
# for k in common_list:
#     w_avg = (net2[k] + net1[k]) / 2.0
#     net2[k] = w_avg
#     net1[k] = w_avg
#
# print(common_list)
#
# print(net1)
# print(net1)
