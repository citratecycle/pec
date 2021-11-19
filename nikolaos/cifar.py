import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from nikolaos.bof_utils import LogisticConvBoF
import utils

# from nn_utils import param_counter

'''
the difference between [CIFAR_NET] and [CIFAR_NET_LIGHT]:
    1.  the former uses concatenation but the latter uses addition
    2.  the latter also includes traditional maxpooling method but 
        the latter does not.
'''

'''
for training [CIFAR_NET_LIGHT], we need to:
    1.  first train network.parameters()
    2.  then train exit parameters by what?
'''
class CIFAR_NET(nn.Module):
    '''
    early-exit:                                  e1                      e2
    layer name:           conv1,   conv2(maxpool),  conv3,  conv4(maxpool), fc1,      fc2
    channel_num:        3,      16,             32,       64,             128,   1024,   10
    channel size:       32,     30,             28(14),   12,             10(5), N/A,    N/A
    '''
    def __init__(self, aggregation='bof'):
        '''
        [aggretation]: 
            [spatial_bof_1]: 2 exit layers, codeword length: 32
            [spatial_bof_2]: 2 exit layers, codeword length: 64
            [spatial_bof_3]: 2 exit layers, codeword length: 256
            for the methods above, "spatial" means the [s] is not directly the average of all
                u_ijk, but instead the average of part of them
            [bof]: 2 exit layers, codeword length: 64, no spatial segmentation is applied
            [elastic]: 2 exit layers, instead of bof, traditional pooling method is applied
        [exit_layer]: 
            -1: the original network output
            -2: the original network output and all 3 exits
            0:  only the first exit layer
            1:  only the second exit layer
            2:  the third exit layer, which is the concatenation of first two exit layers
        '''
        super(CIFAR_NET, self).__init__()

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
            self.exit_2_fc = nn.Linear(4 * 32, 10)

            # Combined exit 1 and exit 2
            self.exit_1_2_fc = nn.Linear(4 * 32 + 4 * 32, 10)

        elif aggregation == 'spatial_bof_2':
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 64, 10)

            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=2)
            self.exit_2_fc = nn.Linear(4 * 64, 10)

            # Combined exit 1 and exit 2
            self.exit_1_2_fc = nn.Linear(4 * 64 + 4 * 64, 10)

        elif aggregation == 'spatial_bof_3':
            self.exit_1 = LogisticConvBoF(32, 256, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 256, 10)

            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 256, split_horizon=2)
            self.exit_2_fc = nn.Linear(4 * 256, 10)

            # Combined exit 1 and exit 2
            self.exit_1_2_fc = nn.Linear(4 * 256 + 4 * 256, 10)

        elif aggregation == 'bof':
            # split_horizon should be 14 ??
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=12)
            self.exit_1_fc = nn.Linear(64, 10)

            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=4)
            self.exit_2_fc = nn.Linear(64, 10)

            # Combined exit 1 and exit 2
            self.exit_1_2_fc = nn.Linear(64 + 64, 10)

        elif aggregation == 'elastic':

            self.exit_1_fc = nn.Linear(32, 10)

            # Exit layer 2:
            self.exit_2_fc = nn.Linear(128, 10)

            # Combined exit 1 and exit 2
            self.exit_1_2_fc = nn.Linear(32 + 128, 10)

        # Flag for switching between layers
        self.exit_layer = -1

        self.elastic_parameters = []
        if 'bof' in self.aggregation:
            self.elastic_parameters.extend(list(self.exit_1.parameters()))
            self.elastic_parameters.extend(list(self.exit_1_fc.parameters()))

            self.elastic_parameters.extend(list(self.exit_2.parameters()))
            self.elastic_parameters.extend(list(self.exit_2_fc.parameters()))

            self.elastic_parameters.extend(list(self.exit_1_2_fc.parameters()))
        else:
            self.elastic_parameters.extend(list(self.exit_1_fc.parameters()))
            self.elastic_parameters.extend(list(self.exit_2_fc.parameters()))
            self.elastic_parameters.extend(list(self.exit_1_2_fc.parameters()))
        # param_counter(self.elastic_parameters)

    def set_exit_layer(self, exit_layer):
        self.exit_layer = exit_layer

    def forward(self, x):
        if self.exit_layer == -1:
            return self.forward_all(x)
        elif self.exit_layer == -2:
            return self.forward_elastic(x, layer=-1)
        elif self.exit_layer >= 0:
            return self.forward_elastic(x, layer=self.exit_layer)


    def forward_all(self, x):

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
        # return F.log_softmax(x, dim=1)
        return x

    def forward_elastic(self, x, layer=-1):
        """
        - 1 means training, 0 is first, 1 is second, 2 is combined
        :param x:
        :param layer:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        if layer in [-1, 0, 2]:
            if 'bof' in self.aggregation:
                x_exit_1 = self.exit_1(x)
            else:
                x_exit_1 = F.avg_pool2d(x, x.size(2))
                x_exit_1 = torch.squeeze(x_exit_1)

            exit1 = self.exit_1_fc(x_exit_1)
            # exit1 = F.log_softmax(exit1, dim=1)
            if layer == 0:
                return exit1

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        if layer in [-1, 1, 2]:
            if 'bof' in self.aggregation:
                x_exit_2 = self.exit_2(x)
            else:
                x_exit_2 = F.avg_pool2d(x, x.size(2))
                x_exit_2 = torch.squeeze(x_exit_2)

            exit2 = self.exit_2_fc(x_exit_2)
            # exit2 = F.log_softmax(exit2, dim=1)
            if layer == 1:
                return exit2

        if layer == -1 or layer == 2:
            x_exit_3 = torch.cat([x_exit_1, x_exit_2], dim=1)
            x_exit_3 = self.exit_1_2_fc(x_exit_3)
            # exit3 = F.log_softmax(x_exit_3, dim=1)
            exit3 = x_exit_3
            if layer == 2:
                return exit3

        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)

        return [exit1, exit2, exit3, x]




class CIFAR_NET_Light(nn.Module):
    def __init__(self, aggregation='bof'):
        super(CIFAR_NET_Light, self).__init__()

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


        # Flag for switching between layers
        self.exit_layer = -1

        self.average_activation_1 = None
        self.average_activation_2 = None
        self.average_activation_combined = None
        self.num_average_1 = 0
        self.num_average_2 = 0
        self.num_average_combined = 0

        # param_counter(self.elastic_parameters)


    def set_exit_layer(self, exit_layer):
        self.exit_layer = exit_layer
    
    def _accumulate_average_activations(self, layer_num, param):
        if layer_num == 1:
            self.num_average_1 += 1
            if self.average_activation_1 is None:
                self.average_activation_1 = utils.calculate_average_activations( param )
            else:
                self.average_activation_1 += utils.calculate_average_activations( param )
        elif layer_num == 2:
            self.num_average_2 += 1
            if self.average_activation_2 is None:
                self.average_activation_2 = utils.calculate_average_activations( param )
            else:
                self.average_activation_2 += utils.calculate_average_activations( param )
        elif layer_num == 3:
            self.num_average_combined += 1
            if self.average_activation_combined is None:
                self.average_activation_combined = utils.calculate_average_activations( param )
            else:
                self.average_activation_combined += utils.calculate_average_activations( param )
        else:
            raise NotImplementedError
        
    def reset_average_activations(self):
        self.average_activation_1, self.average_activation_2, self.average_activation_combined = None, None, None
        self.num_average_1, self.num_average_2, self.num_average_combined = 0, 0, 0

    def print_average_activations(self):
        print1 = (self.average_activation_1 / self.num_average_1) if self.num_average_1 != 0 and self.average_activation_1 is not None else None
        print2 = (self.average_activation_2 / self.num_average_2) if self.num_average_2 != 0 and self.average_activation_2 is not None else None
        print3 = (self.average_activation_combined / self.num_average_combined) if self.num_average_combined != 0 and self.average_activation_combined is not None else None
        print(f'activation 1: {print1} | activation 2: {print2} | activation combined: {print3}')

    def forward(self, x):
        if self.exit_layer == -1:
            return self.forward_all(x)
        elif self.exit_layer == -2:
            return self.forward_elastic(x, layer=-1)
        elif self.exit_layer >= 0:
            return self.forward_elastic(x, layer=self.exit_layer)


    def forward_all(self, x):

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
        # return F.log_softmax(x, dim=1)
        return x

    def forward_elastic(self, x, layer=-1):
        """
        - 1 means training, 0 is first, 1 is second, 2 is combined
        :param x:
        :param layer:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        if layer in [-1, 0, 2]:
            x_exit_1 = self.exit_1(x)
            self._accumulate_average_activations( layer_num=1, param=x_exit_1 )
            exit1 = self.exit_1_fc(x_exit_1)
            # exit1 = F.log_softmax(exit1, dim=1)
            if layer == 0:
                return exit1

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        if layer in [-1, 1, 2]:
            x_exit_2 = self.exit_2(x)
            self._accumulate_average_activations( layer_num=2, param=x_exit_2 )
            exit2 = self.exit_1_fc(x_exit_2)
            # exit2 = F.log_softmax(exit2, dim=1)
            if layer == 1:
                return exit2

        if layer == -1 or layer == 2:
            x_exit_3 = x_exit_1 + x_exit_2
            self._accumulate_average_activations( layer_num=3, param=x_exit_3 )
            x_exit_3 = self.exit_1_fc(x_exit_3)
            # exit3 = F.log_softmax(x_exit_3, dim=1)
            exit3 = x_exit_3
            if layer == 2:
                return exit3

        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)

        return (exit1, exit2, exit3, x)

'''
1. 问宋老师 无痕胶
2. 问宋老师 搬桌子
3. 放音乐
4. 打印 ¥12 for each
'''