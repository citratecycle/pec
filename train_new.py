"""
DESCRIPTION:    this file contains all the functions used for training network with early-exit layers

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import torch

from models_new import get_train_model
import global_param as gp
import utils_new as utils

def train( args ):
    '''
    train the model
    1. get model according to args.model_name
    2. load model according to args.pretrained_file
    3. do the training according to args.train_mode
    4. save model
    '''
    model = get_train_model( args )
    if args.train_mode == 'normal':
        train_normal( args )
    elif args.train_mode == 'original':
        train_original( args )
    elif args.train_mode == 'exits':
        train_exits( args )
    pass

def train_normal( args ):
    '''
    train the model without any early-exit mechanisms, i.e. the most traditional scheme
    '''
    # training setup
    hyper = gp.cifar_normal_train_hyper
    model = get_train_model( args ).to( args.device )
    optimizer = gp.get_optimizer( params=model.parameters(), lr=hyper['learning_rate'] )
    train_loader = gp.get_dataloader( name='cifar_train', train_mode=args.train_mode )
    test_loader = gp.get_dataloader( name='cifar_test', train_mode=args.train_mode )
    # begin training
    best_test_acc = 0
    num_no_increase = 0
    model.train()
    for epoch_idx in range( gp.cifar_normal_train_hyper.epoch_num ):
        if num_no_increase >= 5:
            print( 'early exit is triggered' )
            return
        print(f'\nEpoch: {(epoch_idx + 1)}')
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, ( images, labels ) in enumerate( train_loader ):
            images, labels = images.to( args.device ), labels.to( args.device )
            optimizer.zero_grad()
            outputs = model( images )
            loss = gp.criterion( outputs, labels )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max( 1 )
            total += labels.size( 0 )
            correct += predicted.eq( labels ).sum().item()
            if batch_idx % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.5f |  Acc: %.3f%% (%d/%d)' %
                    (epoch_idx + 1, batch_idx + 1, train_loss / 2000, 100.*correct/total, correct, total))
                train_loss, total, correct = 0.0, 0, 0
        if epoch_idx % 5 == 4:
            print('begin middle test:')
            current_test_acc = test_normal( args )
            # TODO: implement the test_normal function
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                num_no_increase = 0
            else:
                num_no_increase += 1
    test_normal( args )
    torch.save( model, utils.create_model_file_name( args ) )
    # TODO: implement the above function

def test_normal( args ):
    '''
    test the accuracy!
    '''
    # TODO: implement the above funciton
    pass

def train_original( args ):
    '''
    train the model with early-exit structures but only train the original part
    '''
    # TODO: implement the above funciton
    pass

def train_exits( args ):
    '''
    train the early-exits part with the original parts fixed
    '''
    # TODO: implement the above funciton
    pass