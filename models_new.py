"""
DESCRIPTION:    this file contains all the models as well as the methods to call the models

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from nikolaos.bof_utils import LogisticConvBoF
import utils
import global_param as gp

'''
a list of models:
    cifar_exits_train:          cifar with early-exits for training
    cifar_exits_eval:           cifar with early-exits for evaluation
    cifar_normal:               cifar without early-exits
'''


def get_train_model( args ):
    '''
    get the model according to args.model_name and args.train_mode
    '''
    if args.model_name == 'cifar':
        if args.train_mode == 'normal':
            return cifar_normal()
        elif args.train_mode in ['original', 'exits']:
            return cifar_exits_train( args, gp.cifar_exits_train_init )
            # TODO: implement the above class
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid. Should be normal, original or exits' )
            raise NotImplementedError
    elif args.model_name == 'vgg':
        if args.train_mode == 'normal':
            return vgg_normal()
            # TODO: implement the above class
        elif args.train_model in ['original', 'exits']:
            return cifar_exits_train( args, gp.vgg_exits_train_init )
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid. Should be either cifar or vgg' )
        raise NotImplementedError


class cifar_exits_eval(nn.Module):
    def __init__(self, avg_cls_act_list, beta, aggregation='bof'):
        super(cifar_exits_eval, self).__init__()
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


class cifar_exits_train( cifar_exits_eval ):
    '''
    adds functions to specify the exit layers
    to initialize, directly pass in a dict specified by global_param.cifar_exits_train_init
    '''
    pass


class cifar_normal(nn.Module):
    
    def __init__(self):
        super(cifar_normal, self).__init__()
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
    

class vgg_exits_eval( nn.Module ):
    pass


class vgg_exits_train( vgg_exits_eval ):
    pass


class vgg_normal( nn.Module ):
    pass