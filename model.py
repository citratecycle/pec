import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from nikolaos.bof_utils import LogisticConvBoF
import utils

class CIFAR_Adaptive(nn.Module):
    def __init__(self, avg_cls_act_list, beta, aggregation='bof'):
        super(CIFAR_Adaptive, self).__init__()

        self.aggregation = aggregation
        # Base network
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)

        # Exit layer 1:
        if aggregation == 'spatial_bof_1':
            self.exit_1 = LogisticConvBoF(32, 32, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 32, 10)

            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 32, split_horizon=2)

        elif aggregation == 'spatial_bof_2':
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 64, 10)

            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=2)

        elif aggregation == 'spatial_bof_3':
            self.exit_1 = LogisticConvBoF(32, 256, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 256, 10)

            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 256, split_horizon=2)

        elif aggregation == 'bof':
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=12)
            self.exit_1_fc = nn.Linear(64, 10)

            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=4)

        # threshold for switching between layers
        self.activation_threshold_1 = avg_cls_act_list[0]
        self.activation_threshold_2 = avg_cls_act_list[1]
        self.activation_threshold_combined = avg_cls_act_list[2]
        self.average_activation_1 = None
        self.average_activation_2 = None
        self.average_activation_combined = None
        self.num_early_exit_1 = 0
        self.num_early_exit_3 = 0
        self.original = 0

        self.beta = beta
    
    def _accumulate_average_activations(self, layer_num, param):
        if layer_num == 1:
            if self.average_activation_1 is None:
                self.average_activation_1 = utils.calculate_average_activations( param )
            else:
                self.average_activation_1 += utils.calculate_average_activations( param )
        elif layer_num == 2:
            if self.average_activation_2 is None:
                self.average_activation_2 = utils.calculate_average_activations( param )
            else:
                self.average_activation_2 += utils.calculate_average_activations( param )
        elif layer_num == 3:
            if self.average_activation_combined is None:
                self.average_activation_combined = utils.calculate_average_activations( param )
            else:
                self.average_activation_combined += utils.calculate_average_activations( param )
        else:
            raise NotImplementedError
    
    def set_activation_thresholds( self, threshold_list:list ):
        self.activation_threshold_1 = threshold_list[0]
        self.activation_threshold_2 = threshold_list[1]
        self.activation_threshold_combined = threshold_list[2]
    
    def set_beta( self, beta ):
        self.beta = beta

    def forward( self, x ):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x_exit_1 = self.exit_1(x)
        self._accumulate_average_activations( 1, x_exit_1 )
        if self._calculate_average_activation( x ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            exit1 = self.exit_1_fc(x_exit_1)
            return exit1

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x_exit_2 = self.exit_2(x)
        self._accumulate_average_activations( 2, x_exit_2 )
        if self._calculate_average_activation( x ) > self.beta * self.activation_threshold_2:
            self.num_early_exit_3 += 1
            x_exit_3 = (x_exit_1 + x_exit_2)
            x_exit_3 = self.exit_1_fc(x_exit_3)
            # exit3 = F.log_softmax(x_exit_3, dim=1)
            exit3 = x_exit_3
            return exit3

        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        self.original += 1
        return x


class CIFAR_Normal(nn.Module):
    
    def __init__(self):
        super(CIFAR_Normal, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x