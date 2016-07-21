import numpy as np

def ReLU(matrix):
    rowNum,colNum=matrix.shape[0],matrix.shape[1]
    for row in range(rowNum):
        for col in range(colNum):
            if matrix[row,col] < 0.0:
                matrix[row,col]=0.0
#    row,col=matrix.shape[0],matrix.shape[1]
#    size=row*col
#    matrix=matrix.flatten()
#    matrix=list(map(lambda x: x if x >= 0.0 else 0.0,matrix))
#    matrix=np.matrix(matrix).reshape((row,col))
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
