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


def ReLU_diff(matrix,label=None):
    '''
    Description:
        1     ,if x >= 0.0
        0.0   ,if x < 0.0
    '''

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.switch(X < 0.0,0.0,1))
    return f(matrix)


def sigmoid(matrix):
    '''
    Description: 1/(1+e**(-x))
    '''

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],1/(1+theano.tensor.exp((-1)*X)))
    return f(matrix)


def sigmoid_diff(matrix,label=None):
    '''
    Description: e**(-x)/(1+e**(-x))**2
    '''

    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.exp((-1)*X)/((1+theano.tensor.exp((-1)*X))**2))
    return f(matrix)


def test(matrix):
    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.switch(X < 0.0,0.0,0.5*(X**2)))
    return f(matrix)


def test_diff(matrix,label=None):
    X=theano.tensor.matrix(dtype='float32')
    f=theano.function([X],theano.tensor.switch(X < 0.0,0.0,X))
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


def softMax_diff(matrix,label=None):
    row=np.argmax(label,axis=0)
    col=range(label.shape[1])
    x=np.matrix(np.zeros((label.shape[0],label.shape[1])).astype(dtype='float32'))
    x[row,col]=np.float32(1.0)
    y_r=np.matrix(matrix[row,col])
    y_r=np.repeat(y_r,label.shape[0],axis=0)
    r=(np.float32(-1.0))*np.multiply(y_r,matrix-x)
    return r
