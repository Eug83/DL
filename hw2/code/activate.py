import numpy as np

def ReLU(matrix):
    rowNum,colNum=matrix.shape[0],matrix.shape[1]
    for row in range(rowNum):
        for col in range(colNum):
            matrix[row,col]=np.amax([matrix[row,col],0.0])
    return matrix

def ReLU_diff(matrix):
    rowNum,colNum=matrix.shape[0],matrix.shape[1]
    for row in range(rowNum):
        for col in range(colNum):
            if matrix[row,col] < 0:
                matrix[row,col]=0
            else:
                matrix[row,col]=1
    return matrix
