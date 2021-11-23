"""
DESCRIPTION:    this file contains all the functions to apply adaptive early-exit mechanisms
                during the inference process of the network

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

from os import getegid
import torch
from torch import utils

from models_new import get_eval_model
import global_param as gp
from train_new import test_normal
import utils_new as utils

def get_inference_model( args ):
    '''
    get the model for inference from the pretrained model
    1. initialize a model according to args.model_name
    2. copy the parameters to model for inference 
    3. return the model
    '''
    trained_model = torch.load( args.pretrained_file )
    eval_model = get_eval_model( args )
    eval_state_dict = eval_model.state_dict()
    for name, parameter in trained_model.named_parameters():
        eval_state_dict[name].copy_( parameter )
    eval_model.load_state_dict( eval_state_dict )
    return eval_model.to( args.device )


def inference( args ):
    '''
    conduct the inference
    1. get and load model according to args.pretrained_file
    2. do the test using the functions in train_new.py
    3. save the model
    '''
    model = get_inference_model( args )

    model.eval()
    correct = 0
    total = 0
    test_loader = gp.get_dataloader( args, task='test' )
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to( args.device ), labels.to( args.device )
            outputs = model( images )
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size( 0 )
            correct += ( predicted == labels ).sum().item()
    general_acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)
    model.print_exit_percentage()
    
    torch.save( model, utils.create_model_file_name( args ) )
    pass