import numpy as np

def norm1(r,label):
    return np.mean(np.linalg.norm(r-label,ord=1,axis=0))

def norm1_diff(r,label):
    return np.ones((r.shape[0],r.shape[1]))
