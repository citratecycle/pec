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
            return cifar_exits_train()
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


class cifar_exits_eval( nn.Module ):
    '''
    has two early-exiting options
    '''
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
        self.activation_threshold_combined = avg_cls_act_list[2]
        # the number of activation calculations
        self.num_average_1 = 0
        self.num_average_combined = 0
        # the partially averaged activation values
        self.average_activation_1 = None
        self.average_activation_combined = None
        # the number of early exits
        self.num_early_exit_1 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the hyperparameter that controls the trade-off between accuracy and speed-up
        self.beta = beta
    
    def _accumulate_average_activations(self, layer_num, param):
        if layer_num == 1:
            self.num_average_1 += 1
            if self.average_activation_1 is None:
                self.average_activation_1 = utils.calculate_average_activations( param )
            else:
                self.average_activation_1 += utils.calculate_average_activations( param )
        elif layer_num == 3:
            self.num_average_combined += 1
            if self.average_activation_combined is None:
                self.average_activation_combined = utils.calculate_average_activations( param )
            else:
                self.average_activation_combined += utils.calculate_average_activations( param )
        else:
            print( f'the layer_num ({layer_num}) is invalid, should be 1 or 3' )
            raise NotImplementedError
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != 2:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be 2' )
            raise NotImplementedError
        self.activation_threshold_1 = threshold_list[0]
        self.activation_threshold_combined = threshold_list[1]
    
    def set_beta( self, beta ):
        self.beta = beta

    def _calculate_average_activations( self, layer_num ):
        if layer_num == 1:
            return (self.average_activation_1 / self.num_average_1) \
                    if self.num_average_1 != 0 and \
                        self.average_activation_1 is not None \
                    else None
        elif layer_num == 3:
            return (self.average_activation_combined / self.num_average_combined) \
                    if self.num_average_combined != 0 and \
                        self.average_activation_combined is not None \
                    else None
        else:
            print( f'the layer_num ({layer_num}) is invalid, should be 1 or 3' )
            raise NotImplementedError

    def print_average_activations( self ):
        print1 = self._calculate_average_activations( 1 )
        print3 = self._calculate_average_activations( 3 )
        print(f'activation 1: {print1} | activation combined: {print3}')

    def forward( self, x ):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        self._accumulate_average_activations( 1, x_exit_1 )
        if self._calculate_average_activation( 1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            exit1 = self.exit_1_fc(x_exit_1)
            return exit1
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
        x_exit_3 = (x_exit_1 + x_exit_2) / 2
        self._accumulate_average_activations( 3, x_exit_3 )
        if self._calculate_average_activation( 3 ) > self.beta * self.activation_threshold_combined:
            self.num_early_exit_3 += 1
            exit3 = self.exit_1_fc(x_exit_3)
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
    should have:
    1. the ability to set_exit_layers
    '''
    def __init__( self ):
        super().__init__( [0,0,0], 0, aggregation=gp.cifar_exits_train_init.aggregation )
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 5 * 5 * 128 )
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return x

    def forward_exits( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x_exit_1 = self.exit_1(x)
        # calculate exit1
        self._accumulate_average_activations( layer_num=1, param=x_exit_1 )
        exit1 = self.exit_1_fc(x_exit_1)
        # continue inference
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # calculate exit3
        x_exit_2 = self.exit_2(x)
        x_exit_3 = (x_exit_1 + x_exit_2) / 2
        self._accumulate_average_activations( layer_num=3, param=x_exit_3 )
        exit3 = self.exit_1_fc(x_exit_3)
        return ( exit1, exit3 )


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