#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

void parse_argv(int argc,char** argv,char* (&dataPath),char* (&output),int& batchSize){
	int i=1;
	batchSize=1000;

	while(i<argc){
		if(strcmp(argv[i],"-i")==0){
			dataPath=argv[i+1];
			i += 2;
		}else if(strcmp(argv[i],"-o")==0){
			output=argv[i+1];
			i += 2;
		}else if(strcmp(argv[i],"-batch")==0){
			batchSize=atoi(argv[i+1]);
			i += 2;
		}else{
			printf("Undefined input argument %s\n",argv[i]);
			exit(0);
		}
	}
	return;
}


void get_featDim(char* dataPath,int& featDim,int& dataSize){
	FILE *fp;
	char buf[5000];

	fp=fopen(dataPath,'r');
	while(fgets(buf,5000,fp)!=NULL){
		char* stop=strtok(buf," ");
	}
	return;
}


int main(int argc,char** argv){
	char *dataPath,*output;
	int featDim,dataSize,batchSize;

	parse_argv(argc,argv,dataPath,output,batchSize);
	get_featDim(dataPath,featDim,dataSize);
	printf("%d %d\n",featDim,dataSize);//debug
	return 0;
}
