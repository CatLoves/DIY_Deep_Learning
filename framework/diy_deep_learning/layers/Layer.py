"""
Base class of all neural network layers
"""

import numpy as np 
import math 
import copy 

class Layer(object):
    """
    Base layer class. 

    Pytorch layer list: https://pytorch.org/docs/stable/nn.html  
    """

    @property
    def name(self):
        """
        get name of the layer. 
        For details, refer to https://stackoverflow.com/questions/36367736/use-name-as-attribute 
        """
        return self.__class__.__name__

    def forward_pass(self, X, training):
        """
        process input X forward, the output could be fed into next layer
        """
        raise NotImplementedError 

    def backward_pass(self, accum_grad):
        """
        
        """
        raise NotImplementedError 

