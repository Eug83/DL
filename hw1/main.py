import os
import sys
import dnn
import time
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
    with open(os.path.join(dataPath,'training_data'),'r') as fp:
        line=fp.readline().strip()
        line=line.split(' ')
        featDim=len(line)-2

    count=0
    with open(os.path.join(dataPath,'training_data'),'r') as fp:
        for line in fp:
            count += 1

    return (featDim,count)


def load_phonDict(dataPath):
    d=dict()
    with open(os.path.join(dataPath,'phones/48_39.map'),'r') as fp:
        labelCount=0
        for line in fp:
            line=line.strip().split('\t')
            if line[1] not in d:
                d[line[1]]=labelCount
                labelCount += 1
    return (d,labelCount)


def main():
    dataPath,outFile,batchSize=parse_argv()
    featDim,dataSize=get_featDim(dataPath)
    print('Training data size=%d' % (dataSize))
    phon_dict,labelNum=load_phonDict(dataPath)

    struct_str=str(featDim)+'-128-'+str(labelNum)
    phonNet=dnn.DNN(struct=struct_str,learningRateFunc=0.002)
    count,forwardedDataCount=0,0
    X,y=[],[]
    with open(os.path.join(dataPath,'norm_training_data'),'r') as fp:
        for line in fp:
            line=line.strip().split(' ')
            segId,label,feat=line[0],phon_dict[line[1]],[float(x) for x in line[2:]]
            X.append(feat)
            tmpY=np.zeros((1,labelNum))
            tmpY[0][label]=1.0
            y.append(tmpY)
            count += 1
            if count >= batchSize:
                phonNet.forward(np.transpose(X))
                forwardedDataCount += count
                phonNet.backpropagation(np.transpose(y))
                phonNet.update()
                if (forwardedDataCount/batchSize)%10==0:
                    print('Fowarded %d data' % (forwardedDataCount))
                    print('Cost=%f' % (phonNet.calculate_error(np.transpose(y))))
                count=0
                X,y=[],[]
    phonNet.forward(np.transpose(X))
    forwardedDataCount += count
    print('Fowarded %d data' % (forwardedDataCount))
    print('Finish forwarding all data')

    return


if __name__=='__main__':
    main()
