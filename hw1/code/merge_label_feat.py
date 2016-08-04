'''
Description: divide labels and features into training set and validation set
Parameters:
    -i: data directory; it should contain: phones/48_39.map, label/train.lab, mfcc/train.ark, fbank/train.ark
Output:
    label/train_new.lab,test_new.lab: labels for new training set/validation set
    mfcc/train_new.ark,test_new.ark: mfcc features
    fbank/train_new.ark,test_new.ark: fbank features
Example:
    python3 merge_label_feat.py -i DATA_DIRECTORY
'''

import os
import sys


def parse_argv():
    argv=sys.argv
    dataPath=''
    i=1

    while i < len(argv):
        if argv[i]=='-i':
            dataPath=argv[i+1]
            i += 2
        else:
            print('Undefined input argument %s' % (argv[i]))
            sys.exit(0)

    return dataPath


def load_phonTable(dataPath):
    d=dict()

    with open(os.path.join(dataPath,'phones/48_39.map')) as fp:
        for line in fp:
            line=line.strip().split('\t')
            d[line[0]]=line[1]
    return d


def divide_label(dataPath,phon_dict):
    path=os.path.join(dataPath,'label/train.lab')
    segCount=0
    d=dict()
    with open(path,'r') as fp:
        for line in fp:
            line=line.strip().split(',')
            d[line[0]]=phon_dict[line[1]]
    return d


def divide_feat(dataPath,trainLabel_dict):
    with open(os.path.join(dataPath,'train.ark'),'r') as fp:
        with open(os.path.join(dataPath,'train_new.ark'),'w') as fp_tr:
            train_newline=''
            for line in fp:
                line=line.strip().split(' ')
                segId,feat=line[0],line[1:]
                fp_tr.write('%s%s %s' % (train_newline,segId,trainLabel_dict[segId]))
                for x in feat:
                    fp_tr.write(' %s' % (x))
                train_newline='\n'
    return


def main():
    print('Parsing argument...')
    dataPath=parse_argv()
    print('Loading phone table...')
    phon_dict=load_phonTable(dataPath)
    print('Loading labels from training set...')
    trainLabel_dict=divide_label(dataPath,phon_dict)
#    print('Merging mfcc features with labels...')
#    divide_feat(os.path.join(dataPath,'mfcc'),trainLabel_dict)
    print('Merging fbank features with labels...')
    divide_feat(os.path.join(dataPath,'fbank'),trainLabel_dict)
    return

if __name__=='__main__':
    main()
