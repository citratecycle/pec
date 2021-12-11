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
from create_custom_dataloader import custom_cifar

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model_name', default='vgg19', type=str,
                        help='the model structure', metavar='[cifar, vgg, vgg19]' )
    parser.add_argument( '--pretrained_file', default='models_new/vgg19_exits_update_1.pt', type=str,
                        help='the file name that stores the pre-trained model' )
    parser.add_argument( '--optimizer', default='adam', type=str,
                        help='the optimizer for training', metavar='[sgd, adam]' )
    parser.add_argument( '--train_mode', default='exits', type=str,
                        help='the training mode', metavar='[normal (without exits), original (with exits), exits]' )
    parser.add_argument( '--evaluate_mode', default='exits', type=str,
                        help='the evaluating mode', metavar='[normal (without exits), exits (with exits)]' )
    parser.add_argument( '--task', default='evaluate', type=str,
                        help='to train or to evaluate', metavar='[train, evaluate]' )
    parser.add_argument( '--device', default='cuda', type=str,
                        help='the device on which the model is trained', metavar='[cpu, cuda]' )
    parser.add_argument( '--trained_file_suffix', default='update_2', type=str,
                        help='the suffix added to the name of the file that stores the pre-trained model' )
    parser.add_argument( '--beta', default=9, type=float,
                        help='the coefficient used for accuracy-speed trade-off, the higher the more accurate, range from 0 to 1' )
    parser.add_argument( '--save', default=0, type=int,
                        help='whether or not to save the model. 0 or nonzero' )
    # the following are used to cooperate with hardware control on jetson-tx2
    parser.add_argument( '--core_num', default=2, type=int,
                        help='the number of gpu cores used in inference' )
    parser.add_argument( '--frequency', default=1, type=float,
                        help='the frequency used in inference time' )
    parser.add_argument( '--scene', default='continuous', type=str,
                        help='the application scene. from [continuous, periodical]' )
    parser.add_argument( '--baseline_time', default=1, type=float,
                        help='the time it takes for baseline to finish in periodical scene' )
    parser.add_argument( '--sleep_time', default=1, type=float,
                        help='the time we want the board to sleep in periodical scene, baseline setting' )
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
some urgent test that you need to do:
1.  make sure whether the average activation is taken over the classification outputs or
    the codebook outputs···

probable directions for future improvements:
1.  test whether average activations for correct early-exits are higher than noncorrect early-exits?
2.  test whether average activations for different labels are different?
3.  test whether average activations have a limit when the model converges

1.  train_new 讲函数改成通用化 ☑️
2.  global param 把 exit layers 改成 non-exit layers ☑️
3.  models_new 写一下 vgg，并开始尝试训练
4.  写一下 inference

to add a new model:
1.  add new model in model_new.py
2.  add hyper, init, as well as normal_layer_names in global_param.py
3.  (possibly) modify the train_exit function to balance the exit layers from different stages

to adjust the early-exit layer structures:
1.  modify the model structures in model_new.py


todo: 
1. test whether the 'easy' images are consistently easy among testbenches with different initialization
'''