'''
Description:metrics for evaluating the distance between labels and predictions and its differential
Input:
    r: predictions; m by n 2D-array like matrix where m is the output dimension and n is the number of training examples
    label: labels; format same as above
    nets: neural nets; a list of m by n matrice where m is the number of the neurons in the second layer and n is the number of neurons plus one in the first layer
Return value:
    evaluation metric: average cost over training examples
    differential: differentials; m by n matrix where m is the number of output dimension and n is the number of training examples
'''

import numpy as np

def meanSquare(r,label,nets):
    '''
    Description:mean(0.5*(r_n-label_n)^2)
    '''

    x=0.5*np.linalg.norm(r-label,ord=2,axis=0)
    cost=np.mean(np.multiply(x,x))
    return cost


def meanSquare_diff(r,label,nets):
    '''
    Description:r_n-label_n
    '''

    return r-label


def crossEntropy(r,label,nets):
    return
