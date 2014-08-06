#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include "mrf.h"
using namespace std;
int main(int argc, char *argv[]){
	bool ret=false;
	if(argc==6){
		if(!strcmp(argv[1],"learn")){
			mrf *m;
			m=new mrf();
			m->learn(argv[2],argv[3],atoi(argv[4]),atoi(argv[5]));
			delete m;
			ret=true;
		}
	}else if(argc==7){
		if(!strcmp(argv[1],"test")){
			mrf *m;
			m=new mrf();
			m->test(argv[2],argv[3],argv[4],atoi(argv[5]),atoi(argv[6]));
			delete m;
			ret=true;
		}
	}
	if(ret)
		return 0;
	cout<<"./mrf learn train.mrf model algo iter"<<endl<<"./mrf test model test.mrf result.mrf algo iter"<<endl;
	return 0;
}
