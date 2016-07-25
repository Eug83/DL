'''
Description:activation function and its differential
Input:
    matrix:an 2D-array like matrix
Return value:
    matrix:an 2D-array like matrix after activation
'''

import numpy as np
import theano

def ReLU(matrix):
    '''
    Description:
        x     ,if x >= 0.0
        0.0   ,if x < 0.0
    '''

    matrix=matrix.astype(dtype='float32')

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.switch(X < 0.0,0.0,X))
    return f(matrix)

def ReLU_diff(matrix):
    '''
    Description:
        1     ,if x >= 0.0
        0.0   ,if x < 0.0
    '''

    matrix=matrix.astype(dtype='float32')

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.switch(X < 0.0,0.0,1))
    return f(matrix)
