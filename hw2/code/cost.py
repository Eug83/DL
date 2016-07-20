import numpy as np

def squareEuclDist(r,label,nets):
    x=0.5*np.linalg.norm(r-label,ord=2,axis=0)
    cost=np.mean(np.multiply(x,x))
    return cost

def squareEuclDist_diff(r,label,nets):
    return r-label
