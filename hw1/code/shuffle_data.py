'''
Description: shuffle data
Parameter:
    -i: input data
    -o: output file name
Example:
    python3 shuffle_data.py -i DATA -o OUTPUT
'''

import sys
import random

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


def shuffle(dataPath,output):
    data=[]
    with open(dataPath,'r') as fp:
        for line in fp:
            line=line.strip()
            data.append(line)

    random.shuffle(data)

    newline=''
    with open(output,'w') as fp:
        for x in data:
            fp.write('%s%s' % (newline,x))
            newline='\n'
    return


def main():
    dataPath,output=parse_argv()
    shuffle(dataPath,output)
    return

if __name__=='__main__':
    main()
