"""
DESCRIPTION:    this file contains all the functions to apply adaptive early-exit mechanisms
                during the inference process of the network

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import time
import torch

import models_new
from models_new import get_eval_model
import global_param as gp
import utils_new as utils
from train_new import test_exits
import create_custom_dataloader as ccd
from create_custom_dataloader import custom_cifar

def get_inference_model( args ):
    '''
    get the model for inference from the pretrained model
    1. initialize a model according to args.model_name
    2. copy the parameters to model for inference 
    3. return the model
    '''
    # load model parameters
    trained_model = torch.load( args.pretrained_file ).to( args.device )
    eval_model = get_eval_model( args ).to( args.device )
    eval_state_dict = eval_model.state_dict()
    for name, parameter in trained_model.state_dict().items():
        if name in eval_state_dict.keys(): eval_state_dict[name].copy_( parameter )
    eval_model.load_state_dict( eval_state_dict )
    # calculate and load average activation thresholds
    if args.evaluate_mode == 'exits':
        average_activation_list = calculate_average_activations( args, trained_model )
        eval_model.set_activation_thresholds( average_activation_list )
        eval_model.set_beta( args.beta )
    return eval_model


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
    data_size = 10000
    # test_loader = gp.get_dataloader( args, task='test' )
    test_loader = ccd.get_dataloader( 'easy' )
    if args.evaluate_mode == 'exits':
        correct_list = [0 for _ in range( model.exit_num + 1 )]
        total_list = [0 for _ in range( model.exit_num + 1 )]
    st_time = time.perf_counter()
    with torch.no_grad():
        for index, data in enumerate( test_loader ): 
            if index > data_size: break
            images, labels = data
            images, labels = images.to( args.device ), labels.to( args.device )
            outputs = model( images )
            if args.evaluate_mode == 'exits':
                exit_layer, outputs = outputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size( 0 )
            correct += ( predicted == labels ).sum().item()
            if args.evaluate_mode == 'exits':
                total_list[exit_layer] += labels.size( 0 )
                correct_list[exit_layer] += ( predicted == labels ).sum().item()
    general_acc = 100 * correct / total
    if args.evaluate_mode == 'exits':
        acc_list = [correct_list[i]/total_list[i] if total_list[i] != 0 else None for i in range( len( correct_list ) )]
    end_time = time.perf_counter()
    print( f'time consumed: {end_time - st_time}' )
    print('Accuracy of the network on the 10000 test images: %d %%' % general_acc)
    if args.evaluate_mode == 'exits':
        for exit_idx in range( len( correct_list ) ):
            if acc_list[exit_idx] != None:
                print( f'exit{str(exit_idx)}: {100*acc_list[exit_idx]: .3f}%', end=' | ' )
            else:
                print( f'exit{str(exit_idx)}: {None}', end=' | ' )
        print( '' )
    if args.evaluate_mode == 'exits': model.print_exit_percentage()
    if args.save: torch.save( model, utils.create_model_file_name( args ) )


def calculate_average_activations( args, model, verbose=True ):
    exit_layer = model.exit_layer
    exit_num = model.exit_num
    average_activation_list = [0 for _ in range( exit_num )]
    average_times_list = [0 for _ in range( exit_num )]
    model.set_exit_layer( 'exits' )
    train_loader = gp.get_dataloader( args, 'train' )
    loop_limit = gp.average_activation_train_size / train_loader.batch_size
    for act_idx, (images, labels) in enumerate( train_loader ):
        images, labels = images.to( args.device ), labels.to( args.device )
        outputs = model( images )
        for exit_idx in range( exit_num ):
            average_times_list[exit_idx] += 1
            average_activation_list[exit_idx] += \
                utils.calculate_average_activations( outputs[exit_idx] )
        if act_idx >= loop_limit:
            break
    average_activation_list = [average_activation_list[i] / average_times_list[i] for i in range( exit_num )]
    if verbose:
        for print_idx in range( exit_num ):
            print( f'average activation {print_idx}: {average_activation_list[print_idx]}' )
    model.set_exit_layer( exit_layer )
    return average_activation_list