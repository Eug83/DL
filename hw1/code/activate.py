'''
Description:activation function ,its differential and last layer output function
Input:
    matrix:an 2D-array like matrix
Return value:
    matrix:an 2D-array like matrix
'''

import numpy as np
import theano
from sklearn import preprocessing

def ReLU(matrix):
    '''
    Description:
        x     ,if x >= 0.0
        0.0   ,if x < 0.0
    '''

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.switch(X < 0.0,0.0,X))
    return f(matrix)


def ReLU_diff(matrix):
    '''
    Description:
        1     ,if x >= 0.0
        0.0   ,if x < 0.0
    '''

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.switch(X < 0.0,0.0,1))
    return f(matrix)


def softMax(matrix):
    '''
    Description:
        standardize->softmax
    '''

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.exp(X))
    matrix=preprocessing.scale(matrix)
    matrix=f(matrix)
    r=matrix/np.sum(matrix,axis=0)
    return r
