import os
import sys
import numpy as np
from sklearn import preprocessing

def parse_argv():
    argv=sys.argv
    i=1
    dataPath,output='',''

    while i < len(argv):
        if argv[i]=='-i':
            dataPath=argv[i+1]
            i += 2
        elif argv[i]=='-o':
            output=argv[i+1]
            i += 2
        else:
            print('Undefined input argument %s' % (argv[i]))
            sys.exit(0)
    return (dataPath,output)


def normalize(dataPath,output):
    segId,label,data=[],[],[]
    with open(os.path.join(dataPath,'training_data'),'r') as fp:
        for line in fp:
            line=line.strip().split(' ')
            segId.append(line[0])
            label.append(line[1])
            feat=[float(x) for x in line[2:]]
            data.append(feat)
    
    data=preprocessing.scale(data)

    newline=''
    with open(os.path.join(dataPath,output),'w') as fp:
        for i in range(len(segId)):
            fp.write('%s%s %s' % (newline,segId[i],label[i]))
            for x in data[i]:
                fp.write(' %f' % (x))
            newline='\n'
    return


def main():
    dataPath,output=parse_argv()
    normalize(dataPath,output)
    return

if __name__=='__main__':
    main()
