import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import utils_new as utils

##############################################################################
###                            GENERAL CONSTANTS                           ###
##############################################################################

classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

# cifar_early_exit_names = ['exit_1.a', 'exit_1.c', 'exit_1.n1', 'exit_1.n2', 'exit_1.codebook.weight', 
#                           'exit_1.codebook.bias', 'exit_1_fc.weight', 'exit_1_fc.bias', 'exit_2.a', 
#                           'exit_2.c', 'exit_2.n1', 'exit_2.n2', 'exit_2.codebook.weight', 'exit_2.codebook.bias']

cifar_normal_layer_names = [
    'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 
    'conv4.weight', 'conv4.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'
]

vgg_normal_layer_names = [
    #TODO fill in it
]

##############################################################################
###                                 METHOD                                 ###
##############################################################################

criterion = nn.CrossEntropyLoss()

def get_optimizer( params, lr, op_type ):
    if op_type == 'adam':
        return optim.Adam( params, lr=lr )
    elif op_type == 'sgd':
        return optim.SGD( params, lr=lr )
    else:
        print( f'Error: the optimizer type ({op_type}) is not valid. Should be adam or sgd' )
        raise NotImplementedError

def get_dataloader( args, task ):
    if args.model_name == 'cifar':
        if task == 'train':
            batch_size_dict = {
                "normal": cifar_normal_train_hyper.batch_size, 
                "original": cifar_original_train_hyper.batch_size,
                "exits": cifar_exits_train_hyper.batch_size
            }
            return torch.utils.data.DataLoader( cifar_train_dataset, 
                                                batch_size=batch_size_dict[args.train_mode], 
                                                shuffle=True, 
                                                num_workers=4 )
        elif task == 'test':
            return torch.utils.data.DataLoader( cifar_test_dataset, 
                                                batch_size=1, 
                                                shuffle=True, 
                                                num_workers=4)
    else:
        # TODO: implement vgg dataset
        print( f'Error: dataset name ({args.model_name}) is not valid. Should be cifar' )
        raise NotImplementedError

def get_hyper( args ):
    if args.model_name == 'cifar':
        if args.train_mode == 'normal':
            return cifar_normal_train_hyper
        elif args.train_mode == 'original':
            return cifar_original_train_hyper
        elif args.train_mode == 'exits':
            return cifar_exits_train_hyper
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid, should be normal, original or exits' )
            raise NotImplementedError
    elif args.model_name == 'vgg':
        if args.train_mode == 'normal':
            return vgg_normal_train_hyper
        elif args.train_mode == 'original':
            return vgg_original_train_hyper
        elif args.train_mode == 'exits':
            return vgg_exits_train_hyper
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid, should be normal, original or exits' )
            raise NotImplementedError
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid, should be cifar or vgg' )
        raise NotImplementedError

def get_layer_names( args ):
    if args.model_name == 'cifar':
        return cifar_normal_layer_names
    elif args.model_name == 'vgg':
        return vgg_normal_layer_names
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid, should be cifar or vgg' )
        raise NotImplementedError

##############################################################################
###                                  INIT                                  ###
##############################################################################

cifar_exits_train_init = utils.Namespace(
    aggregation = 'bof'
)

cifar_exits_eval_init = utils.Namespace(
    aggregation = 'bof',
    activation_threshold_list = [
        6.7, 6.0
    ],
    activation_initial_list = [
        12.0, 8.0
    ],
    beta = 2.86      # between 0 and 1, the higher the more accurate
)

vgg_exits_train_init = utils.Namespace(
    # TODO: fill in the dict
)

vgg_exits_eval_init = utils.Namespace(
    # TODO: fill in
)

##############################################################################
###                            HYPER PARAMETERS                            ###
##############################################################################

cifar_normal_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

cifar_original_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

cifar_exits_train_hyper = utils.Namespace(
    epoch_num = 300,
    learning_rate = 0.001,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

vgg_normal_train_hyper = utils.Namespace(

)

vgg_original_train_hyper = utils.Namespace(

)

vgg_exits_train_hyper = utils.Namespace(

)

cifar_train_dataset = torchvision.datasets.CIFAR10( root='./data/cifar10/', 
                                                    train=True, 
                                                    transform=cifar_normal_train_hyper.transform, 
                                                    download=True )

cifar_test_dataset = torchvision.datasets.CIFAR10(  root='./data/cifar10/', 
                                                    train=False, 
                                                    transform=cifar_normal_train_hyper.transform, 
                                                    download=False )

