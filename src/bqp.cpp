#include "bqp.h"
#include "sap.h"
#include <vector>
using namespace std;
double bqp(int n, int m, int *is, int *js, double *a, double *b, bool *x){
/*pseudo boolean quadratic programming:
	max \sum_{ij} a_{ij} x_ix_j + \sum_i b_i x_i
	s.t. x_i \in {0,1}
	where a_{ij}>=0;
is equivalent to
	min 0.5*\sum_{ij}(x_iy_j + y_ix_j) + \sum_i w_i^+ x_i + \sum_i w_i^- y_i
	s.t. x_i,y_i \in {0,1}, x_i + y_i = 1
	where w_i = - b_i - 0.5*\sum_j (a_{ij} + a_{ji})

	call graph_cut
*/
	//adjust a'_{ij}==a'_{ji}=0.5*a_{i,j}
	vector<int> from;
	vector<int> to;
	vector<double> cap1;
	vector<double> cap2;
	int N=n+2;//add source and target
	vector<double> w(n,0);
	double cost=0;//constant for w_i^-
	for(int i=0;i<n;i++)
		w[i]=-b[i];
	for(int i=0;i<m;i++){
		//is->js
		w[is[i]]-=a[i]*0.5;
		//js->is
		w[js[i]]-=a[i]*0.5;
	}
	for(int i=0;i<n;i++)
		if(w[i]<0)
			cost+=w[i];
	//source to node
	for(int i=0;i<n;i++){//0->i+1
		from.push_back(0);
		to.push_back(i+1);
		if(w[i]>=0)
			cap1.push_back(w[i]);
		else
			cap1.push_back(0);
		cap2.push_back(0);
	}
	//node to target
	for(int i=0;i<n;i++){//i+1->N-1
		from.push_back(i+1);
		to.push_back(N-1);
		if(w[i]>=0)
			cap1.push_back(0);
		else
			cap1.push_back(-w[i]);
		cap2.push_back(0);
	}
	for(int i=0;i<m;i++){
		int u=is[i]+1;
		int v=js[i]+1;
		double c=a[i]*0.5;//u->v c, v->u c
		from.push_back(u);
		to.push_back(v);
		cap1.push_back(c);
		cap2.push_back(c);
	}

	int M=from.size();
	bool *cut=new bool[N];
	double maxflow=sap(N,M,&from[0],&to[0],&cap1[0],&cap2[0],cut);
	for(int i=0;i<n;i++)
		x[i]=!cut[i+1];
	delete [] cut;
	return -(maxflow+cost);
}
