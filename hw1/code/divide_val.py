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
    trainLabel_dict,testLabel_dict=dict(),dict()
    with open(path,'r') as fp:
        for line in fp:
            segCount += 1

    with open(path,'r') as fp:
        count=0
        with open(os.path.join(dataPath,'label/train_new.lab'),'w') as fp_tr:
            with open(os.path.join(dataPath,'label/test_new.lab'),'w') as fp_te:
                train_newline,test_newline='',''
                for line in fp:
                    line=line.strip().split(',')
                    if count <= int(segCount*0.9):
                        fp_tr.write('%s%s,%s' % (train_newline,line[0],phon_dict[line[1]]))
                        trainLabel_dict[line[0]]=phon_dict[line[1]]
                        train_newline='\n'
                    else:
                        fp_te.write('%s%s,%s\n' % (test_newline,line[0],phon_dict[line[1]]))
                        testLabel_dict[line[0]]=phon_dict[line[1]]
                        test_newline='\n'
                    count += 1
    return (trainLabel_dict,testLabel_dict)


def divide_feat(dataPath,trainLabel_dict,testLabel_dict):

    with open(os.path.join(dataPath,'train.ark'),'r') as fp:
        with open(os.path.join(dataPath,'train_new.ark'),'w') as fp_tr:
            with open(os.path.join(dataPath,'test_new.ark'),'w') as fp_te:
                train_newline,test_newline='',''
                for line in fp:
                    line=line.strip().split(' ')
                    segId,feat=line[0],line[1:]
                    if segId in trainLabel_dict:
                        fp_tr.write('%s%s %s' % (train_newline,segId,trainLabel_dict[segId]))
                        for x in feat:
                            fp_tr.write(' %s' % (x))
                        train_newline='\n'
                    elif segId in testLabel_dict:
                        fp_te.write('%s%s %s' % (test_newline,segId,testLabel_dict[segId]))
                        for x in feat:
                            fp_te.write(' %s' % (x))
                        test_newline='\n'
                    else:
                        print('Uncategorized segment Id %s' % (segId))
    return


def main():

    print('Parsing argument...')
    dataPath=parse_argv()
    print('Loading phone table...')
    phon_dict=load_phonTable(dataPath)
    print('Dividing labels to training set and testing set...')
    trainLabel_dict,testLabel_dict=divide_label(dataPath,phon_dict)
    print('Dividing mfcc to training set and testing set...')
    divide_feat(os.path.join(dataPath,'mfcc'),trainLabel_dict,testLabel_dict)
    print('Dividing fbank to training set and testing set...')
    divide_feat(os.path.join(dataPath,'fbank'),trainLabel_dict,testLabel_dict)
    return

if __name__=='__main__':
    main()
