/********************************************************
 * this is an implementation of Improved Shortest Augmented Path algorithm (SAP) for Max-Flow/Min Cut
 * Author: Xian Qian, Mail: qx@hlt.utdallas.edu
 * Licence: LGPL
 *********************************************************/
#include "sap.h"
#include <cmath>
#include <vector>
#include <ctime>
#include <cstring>
#include <iostream>
using namespace std;
namespace{
struct edge{
	int e;
	double f;
	edge *next;
};

double INF=1e+37;
double eps=1e-15;

edge * OPT(edge * x,edge *epool){
	return epool+((x-epool)^1);
}

bool positive(double f){
	return f>eps;
}
bool negative(double f){
	return f<-eps;
}
bool zero(double f){
	return (f<=eps && f>=-eps);
}
bool nozero(double f){
	return (f>eps || f<-eps);
}
}//end namespace
double sap(int n, int m, int *from, int *to, double *cap1, double *cap2, bool *cut){
	//cut[i]==1 if node i is connect to source, 0 otherwise
	int N;//0: source , N-1 target
	int M;//edge number
	edge **E;//E[i] is the first edge out from i
	edge *epool;//epool is the storage of edges
	edge *etop;//etop
	N=n;
	M=m;
	E=new edge* [N];
	for (int i=0;i<N;i++)E[i]=NULL;
	epool=new edge[M*2];
	etop=epool;
	for(int i=0;i<M;i++){
		int v=to[i];
		int u=from[i];
		etop->e=v;etop->f=cap1[i];etop->next=E[u];E[u]=etop;etop++;
		etop->e=u;etop->f=cap2[i];etop->next=E[v];E[v]=etop;etop++;
	}
	int *d=new int[N];
	int *g=new int[N+1];
	int *Q=new int[N];
	memset(d,0,sizeof(int)*N);
	memset(g,0,sizeof(int)*N+1);
	memset(Q,0,sizeof(int)*N);
	edge **c=new edge* [N];
	edge **pre=new edge* [N];
	memset(c,0,sizeof(edge*)*N);
	memset(pre,0,sizeof(edge*)*N);

	double ret=0;
	int x=0,bot;
	for (int i=0;i<N;i++)c[i]=E[i];

	pre[0]=0;
	bot=1;
	Q[0]=N-1;
	for (int i=0;i<bot;i++)for (edge *p=E[Q[i]];p;p=p->next)
		if (nozero(OPT(p,epool)->f) && p->e!=N-1 && d[p->e]==0)d[Q[bot++]=p->e]=d[Q[i]]+1;
	for (int i=0;i<N;i++)g[d[i]]++;
	while (d[0]<N){
		while (c[x] && (zero(c[x]->f) || d[c[x]->e]+1!=d[x]))c[x]=c[x]->next;
		if (c[x]){
			pre[c[x]->e]=OPT(c[x],epool);
			x=c[x]->e;
			if (x==N-1){
				double t=INF;
				for (edge *p=pre[N-1];p;p=pre[p->e])if (t>OPT(p,epool)->f)t=OPT(p,epool)->f;
				for (edge *p=pre[N-1];p;p=pre[p->e])
					p->f+=t,OPT(p,epool)->f-=t;
				ret+=t;
				x=0;
			}
		}else{
			int od=d[x];
			g[d[x]]--;
			d[x]=N;
			for (edge *p=c[x]=E[x];p;p=p->next)if (nozero(p->f) && d[x]>d[p->e]+1)d[x]=d[p->e]+1;
			g[d[x]]++;
			if (x)x=pre[x]->e;
			if (!g[od])break;
		}
	}
	for(int i=0;i<N;i++)
		cut[i]=false;
	cut[0]=1;
	Q[0]=0;bot=1;
	for (int i=0;i<bot;i++)
		for (edge *p=E[Q[i]];p;p=p->next){
			if (positive(p->f) && cut[p->e]==false){
				cut[p->e]=true;
				Q[bot++]=p->e;
			}
		}
	delete [] d;
	delete [] g;
	delete [] Q;
	delete [] c;
	delete [] pre;
	delete [] E;
	delete [] epool;
	return ret;
}
