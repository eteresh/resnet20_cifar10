from itertools import chain

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d


class LayerPair(nn.Module):
    """ Base unit of resnet: two convolutional layers transform input vector,
        then input vector added to transformed output vector.
    """
    def __init__(self, n_filters=16, reduce_image_by_half=False):
        super().__init__()
        if reduce_image_by_half:
            self.first_layer = Conv2d(n_filters // 2, n_filters, kernel_size=3, stride=2, padding=1)
        else:
            self.first_layer = Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.second_layer = Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.first_batch_norm = nn.BatchNorm2d(n_filters)
        self.second_batch_norm = nn.BatchNorm2d(n_filters)
        if reduce_image_by_half:
            # input vector and transformed output vector have different dimensions,
            # so we are using convolutional layer just to adopt input vector dimensions
            self.shortcut = Conv2d(n_filters // 2, n_filters, kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = None

        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        self.output_relu = nn.ReLU()

    def forward(self, x):
        y = self.first_layer(x)
        y = self.first_batch_norm(y)
        y = self.relu(y)
        y = self.second_layer(y)
        y = self.second_batch_norm(y)
        if self.shortcut is not None:
            x = self.shortcut(x)
        y = y + x
        return self.output_relu(y)


class ResNet20(nn.Module):
    """ Realization of ResNet20 model, described here https://arxiv.org/pdf/1512.03385.pdf,
    explained here: http://pabloruizruiz10.com/resources/CNNs/ResNet-on-CIFAR10.pdf
    Technical details here: https://keras.io/zh/examples/cifar10_resnet/
    and here: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """
    def __init__(self, n_classes=10):
        super().__init__()

        n_filters = 16
        self.stage_0_transformation_layer = Conv2d(3, n_filters, kernel_size=3, stride=1, padding=1)

        # (32, 32), 16 filters
        self.stage_0 = nn.Sequential(*(LayerPair(n_filters=n_filters) for _ in range(3)))

        # (16, 16), 32 filters
        n_filters = 32
        self.stage_1 = nn.Sequential(*(LayerPair(n_filters=n_filters, reduce_image_by_half=(idx == 0))
                                       for idx in range(3)))

        # (8, 8), 64 filters
        n_filters = 64
        self.stage_2 = nn.Sequential(*(LayerPair(n_filters=n_filters, reduce_image_by_half=(idx == 0))
                                       for idx in range(3)))

        self.output = nn.Linear(n_filters, n_classes)

    def forward(self, x):
        x = self.stage_0_transformation_layer(x)
        x = self.stage_0(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, 64)
        return self.output(x)

    def get_all_convolutional_layers(self):
        return list(chain(
            (self.stage_0_transformation_layer,),
            *[(layer_pair.first_layer, layer_pair.second_layer) for layer_pair in chain(
                self.stage_0,
                self.stage_1,
                self.stage_2
            )]
        ))
