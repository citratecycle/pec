from argparse import Namespace
from models_new import cifar_exits_train

import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import utils_new as utils

criterion = nn.CrossEntropyLoss()

cifar_exits_train_init = utils.Namespace(
    # TODO: fill in the dict
)

vgg_exits_train_init = utils.Namespace(
    # TODO: fill in the dict
)

cifar_normal_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

cifar_train_dataset = torchvision.datasets.CIFAR10( root='./data/cifar10/', 
                                                    train=True, 
                                                    transform=cifar_normal_train_hyper.transform, 
                                                    download=True )

cifar_test_dataset = torchvision.datasets.CIFAR10(  root='./data/cifar10/', 
                                                    train=False, 
                                                    transform=cifar_normal_train_hyper.transform, 
                                                    download=False)

def get_optimizer( params, lr, op_type ):
    if op_type == 'adam':
        return optim.Adam(params, lr=lr)
    elif op_type == 'sgd':
        return optim.SGD( params, lr=lr )
    else:
        print( f'Error: the optimizer type ({op_type}) is not valid. Should be adam or sgd' )
        raise NotImplementedError

def get_dataloader( name, train_mode ):
    if name == 'cifar_train':
        batch_size_dict = {
            "normal": cifar_normal_train_hyper.batch_size, 
            "original": cifar_original_train_hyper.batch_size,
            # TODO: implement the above namespace
            "exits": cifar_exits_train_hyper.batch_size
            # TODO: implement the above namespace
        }
        return torch.utils.data.DataLoader( cifar_train_dataset, 
                                            batch_size=batch_size_dict[train_mode], 
                                            shuffle=True, 
                                            num_workers=4 )
    elif name == 'cifar_test':
        return torch.utils.data.DataLoader( cifar_test_dataset, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=4)
    else:
        print( f'Error: dataset name ({name}) is not valid. Should be cifar_train or cifar_test' )
        raise NotImplementedError