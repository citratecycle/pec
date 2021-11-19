"""
DESCRIPTION:    this file contains the main program

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import argparse
import torch

from inference_new import inference
from train_new import train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model_name', default='cifar', type=str,
                        help='the model structure', metavar='[cifar, vgg]' )
    parser.add_argument( '--pretrained_file', default='models/trained_all.pt', type=str,
                        help='the file name that stores the pre-trained model' )
    parser.add_argument( '--early_exit_json', default='input_json/default.json', type=str,
                        help='the json file name that specify the early-exit structures' )
    parser.add_argument( '--optimizer', default='sgd', type=str,
                        help='the optimizer for training', metavar='[sgd, adam]' )
    parser.add_argument( '--train_mode', default='original', type=str,
                        help='the training mode', metavar='[normal (without exits), original (with exits), exits]' )
    parser.add_argument( '--task', default='train', type=str,
                        help='to train or to evaluate', metavar='[train, evaluate]' )
    parser.add_argument( '--device', default='cpu', type=str,
                        help='the device on which the model is trained', metavar='[cpu, cuda]' )
    args = parser.parse_args()
    args.torch_device = torch.device( 'cpu' ) if args.device == 'cpu' else torch.device( 'cuda' )
    return args

if __name__ == '__main__':
    args = get_args()
    print( f'current device is {torch.cuda.current_device()}' )
    if args.task == 'evaluate':
        inference( args )
    elif args.task == 'train':
        train( args )
    else:
        print( f'Error: args.task ({args.task}) is not valid. Should be either train or evaluate' )
        raise NotImplementedError
    



'''
1. train_new 讲函数改成通用化 ☑️
2. global param 把 exit layers 改成 non-exit layers
3. models_new 写一下 vgg，并开始尝试训练
4. 写一下 inference
'''