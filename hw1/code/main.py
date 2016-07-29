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
import numpy as np
import shuffle_data

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
    phon48_dict,phon39_dict,phon_map=dict(),dict(),dict()
    with open(os.path.join(dataPath,'phones/48_39.map'),'r') as fp:
        label48Count,label39Count=0,0
        for line in fp:
            line=line.strip().split('\t')
            phon48_dict[line[0]]=label48Count
            label48Count += 1
            if line[1] not in phon39_dict:
                phon39_dict[line[1]]=label39Count
                label39Count += 1
            phon_map[phon48_dict[line[0]]]=phon39_dict[line[1]]
    return (phon48_dict,phon39_dict,phon_map,label48Count)


def train(phonNet,phonCount_dict,dataPath,batchSize,labelNum):
    count,forwardedDataCount=0,0
    X,y=[],[]

    trainDataPath=os.path.join(dataPath,'norm_training_data')
    shuffle_data.shuffle(trainDataPath,trainDataPath)

    with open(trainDataPath,'r') as fp:
        for line in fp:
            line=line.strip().split(' ')
            segId,label,feat=line[0],phonCount_dict[line[1]],[float(x) for x in line[2:]]
            X.append(feat)
            tmpY=[0]*labelNum
            tmpY[label]=1.0
            y.append(tmpY)
            count += 1
            if count >= batchSize:
                X,y=np.matrix(X).astype(dtype='float32'),np.matrix(y).astype(dtype='float32')
                phonNet.forward(np.transpose(X))
                forwardedDataCount += count
                if (forwardedDataCount/batchSize)%10==0:
                    print('Fowarded %d data' % (forwardedDataCount))
                    print('Cost=%f' % (phonNet.calculate_error(np.transpose(y))))
                phonNet.backpropagation(np.transpose(y))
                phonNet.update()
                count=0
                X,y=[],[]
    if count >= 1:
        phonNet.forward(np.transpose(X))
        forwardedDataCount += count
    print('Fowarded %d data' % (forwardedDataCount))
    print('Finish forwarding all data')
    return phonNet


def test(phonNet,phon48_dict,phon_map,dataPath,batchSize,labelNum):
    correct,totalCount=0,0
    count=0
    X,y=[],[]

    with open(os.path.join(dataPath,'norm_testing_data'),'r') as fp:
        for line in fp:
            line=line.strip().split(' ')
            segId,label,feat=line[0],phon_map[phon48_dict[line[1]]],[float(x) for x in line[2:]]
            X.append(feat)
            y.append(label)
            count += 1
            if count >= batchSize:
                X=np.matrix(X)
                r=phonNet.predict(np.transpose(X))
                for i in range(len(r)):
                    r[i]=phon_map[r[i]]
                tmpCor,tmpTotal=phonNet.score(np.array(r),np.array(y))
                correct += tmpCor
                totalCount += tmpTotal
                X,y=[],[]
                count=0
    accuracy=float(correct)/float(totalCount)
    print('Accuracy=%f' % (accuracy))
    return (phonNet,accuracy)


def main():
    dataPath,outFile,batchSize=parse_argv()
    featDim,dataSize=get_featDim(dataPath)
    print('Training data size=%d' % (dataSize))
    phon48_dict,phon39_dict,phon_map,labelNum=load_phonDict(dataPath)

    struct_str=str(featDim)+'-128-'+str(labelNum)
    learningRate_str=0.025
    actiFunc_str='ReLU'
    momentum_float=0.4
    lastAccu,accu=0.0,0.0
    epochCount=1
    phonNet=dnn.DNN(struct=struct_str,learningRateFunc=learningRate_str,actiFunc=actiFunc_str,momentum=momentum_float)
    phonNet.load_model()
    print('Training dnn...')
    while True:
        print('-------------------------------------------')
        print('round %d' % (epochCount))
        phonNet=train(phonNet,phon48_dict,dataPath,batchSize,labelNum)
        print('Testing...')
        phonNet,accu=test(phonNet,phon48_dict,phon_map,dataPath,batchSize,labelNum)
        if accu > lastAccu:
            lastAccu=accu
        else:
            break
        epochCount += 1
        phonNet.save_model()
        print('-------------------------------------------')
    print('Accuracy=%f' % (accu))
    return


if __name__=='__main__':
    main()
