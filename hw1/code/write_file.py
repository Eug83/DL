'''
Description: DL hw1
Parameter:
    -i: input data directory; it should contain:
        norm_training_data: normalized training data
        norm_testing_data: normalized testing data
        phones/48_39.map: phones map
    -o: output file name
    -batch: batch size; default=1000
Example:
    python3 main.py -i DATA_DIRECTORY -o OUTPUT -batch 1000
'''

import os
import sys
import dnn
import time
import copy
import numpy as np

def parse_argv():
    argv=sys.argv
    i=1
    dataPath,outFile='',''
    batchSize=1000

    while i < len(argv):
        if argv[i]=='-i':
            dataPath=argv[i+1]
            i += 2
        elif argv[i]=='-o':
            outFile=argv[i+1]
            i += 2
        elif argv=='-batch':
            batchSize=int(argv[i+1])
            i += 2
        else:
            print('Undefined input argument: %s' % (argv[i]))
            sys.exit(0)

    return (dataPath,outFile,batchSize)


def get_featDim(dataPath):
    featDim=0
    with open(os.path.join(dataPath,'norm_training_data'),'r') as fp:
        line=fp.readline().strip()
        line=line.split(' ')
        featDim=len(line)-2

    count=0
    with open(os.path.join(dataPath,'norm_training_data'),'r') as fp:
        for line in fp:
            count += 1

    return (featDim,count)


def load_phonDict(dataPath):
    phon48_dict,phon_map=dict(),dict()
    with open(os.path.join(dataPath,'phones/48_39.map'),'r') as fp:
        label48Count=0
        for line in fp:
            line=line.strip().split('\t')
            phon48_dict[line[0]]=label48Count
            label48Count += 1
            phon_map[phon48_dict[line[0]]]=line[1]
    return (phon48_dict,phon_map,label48Count)


def test(phonNet,phon48_dict,phon_map,dataPath,outFile,batchSize,labelNum):
    correct,totalCount=0,0
    count=0
    X,idList=[],[]
    newline=''

    with open(os.path.join(dataPath,'test_data'),'r') as fp:
        with open(outFile,'w') as fp_write:
            fp_write.write('Id,Prediction\n')
            for line in fp:
                line=line.strip().split(' ')
                segId,feat=line[0],[float(x) for x in line[1:]]
                idList.append(segId)
                X.append(feat)
                count += 1
                if count >= batchSize:
                    X=np.matrix(X).astype(dtype='float32')
                    r=phonNet.predict(np.transpose(X))
                    for i in range(len(r)):
                        fp_write.write('%s%s,%s' % (newline,idList[i],phon_map[r[i]]))
                        newline='\n'
                    X,idList=[],[]
                    count=0
            X=np.matrix(X).astype(dtype='float32')
            r=phonNet.predict(np.transpose(X))
            for i in range(len(r)):
                fp_write.write('%s%s,%s' % (newline,idList[i],phon_map[r[i]]))
    return


def main():
    dataPath,outFile,batchSize=parse_argv()
    featDim,dataSize=get_featDim(dataPath)
    phon48_dict,phon_map,labelNum=load_phonDict(dataPath)

    struct_str=str(featDim)+'-128-'+str(labelNum)
    learningRate_str=0.025
    actiFunc_str='ReLU'
    momentum_float=0.4
    lastAccu,accu=0.0,0.0
    epochCount=1
    phonNet=dnn.DNN(struct=struct_str,learningRateFunc=learningRate_str,actiFunc=actiFunc_str,momentum=momentum_float)
    phonNet.load_model()
    test(phonNet,phon48_dict,phon_map,dataPath,outFile,batchSize,labelNum)
    return


if __name__=='__main__':
    main()
