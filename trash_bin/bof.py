"""
DESCRIPTION:    this file contains all the APIs that will be used in Bag of Features algorithm

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import torch

from utils import Namespace

class bof_handler:
    '''
    the handler for bof algorithm, which possesses the methods to generate codebooks, 
    feature vectors and so on.
    '''
    def __init__( args:Namespace ):
        '''
        
        '''
        pass
    
    def _get_features( self, parameters:torch.tensor ):
        '''
        given the parameters at a certain layer, return a list containing all the features
        '''
        # tell the layer type of the given parameter
        layer_shape = parameters.shape
        is_conv =   True if layer_shape >  2 else False
        is_fc =     True if layer_shape == 2 else False
        is_bias =   True if layer_shape == 1 else False
        # extract all the features only for convolutional layers
        if is_fc:
            raise Exception('you cannot extract features from fully-connected layers')
        elif is_bias:
            raise Exception('you cannot extract features from bias layers')
        elif is_conv:
            # the shape is (next_num) x (cur_num) x (ker_size) x (ker_size)
            
            pass
        pass

    def generate_codebook( self, features:list, args:Namespace ):
        '''
        given the list of features, generate a codebook either based on randomly selection
        or by using clustering algorithms
        '''
        pass
    
    def generate_feature_vector( self ):
        '''
        
        '''
        pass