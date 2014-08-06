#include "bqp.h"
#include "mrf.h"
#include <set>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace std;
const int MAXSTRLEN=1024*1024;
const double INF=1e+20;
const double EPS=1e-10;
bool split_string(char *str, const char *cut, vector<char *> &strs)
{
	char *p=str,*q;
	bool ret=false;
	strs.clear();
	while (q=strstr(p,cut))
	{
		*q=0;
		strs.push_back(p);
		p=q+strlen(cut);
		ret=true;
	}
	if(p) strs.push_back(p);
	return ret;
}
mrf::mrf(){
	_algo=0;
	_max_iter=5000;
	_dual_iter=10;
}
mrf::~mrf(){
	_graphs.clear();
}
bool mrf::learn(char *train_file, char *model_file, int algo, int dual_iter){
	_algo=algo;
	_dual_iter=dual_iter;
	//read data
	int lambda_sz=read_data(train_file);
	if(lambda_sz==0)
		return false;
	_lambda.resize(lambda_sz,0);
//	_lambda[0]=-1;
//	_lambda[1]=-1;
//	_lambda[2]=-1;
//	_lambda[3]=4;
//	_lambda[4]=5;
//	_lambda[5]=6;
//	_lambda[6]=10;
	_sum=_lambda;
	int time=_max_iter*_graphs.size();
	double total_p=0;
	double total_d=0;
	int total_iter=0;
	for(int iter=0;iter<_max_iter;iter++){
		int total=0;
		int err=0;
		for(int i=0;i<_graphs.size();i++){
			graph &g=_graphs[i];
			vector<int> y;
			cal_score(g);
			bool optimal=true;
			double p,d;
			int act_iter=0;
			if(_algo==0)
				optimal=decode(g,y,p,d,act_iter);
			else if(_algo==1)
				optimal=decode_naive_dd(g,y,p,d,act_iter);
			else
				decode_gurobi(g,y);
			err+=update(g,y,time);
			total_p+=p;
			total_d+=d;
			total_iter+=act_iter;
//			cout<<err<<endl;
			total+=y.size();
			time--;
		}
		cout<<"iter: "<<iter<<"\terr: "<<err<<'/'<<total<<"="<<double(err)/total<<endl;
	}
	//average
	for(int i=0;i<_lambda.size();i++)
		_lambda[i]=_sum[i]/(_max_iter*_graphs.size());
	save_model(model_file);
	return true;
}
bool mrf::test(char *model_file, char *test_file, char *result_file,int algo, int dual_iter){
	_dual_iter=dual_iter;
	_algo=algo;
	int lambda_sz=read_data(test_file);
	if(lambda_sz==0)
		return false;
	if(!load_model(model_file))
		return false;
	int err=0;
	int total=0;
	ofstream ofs;
	ofs.open(result_file);
	int opts=0;

	double total_p=0;
	double total_d=0;

	double timecost=0;
	int total_iter=0;
	for(int i=0;i<_graphs.size();i++){
		graph &g=_graphs[i];

		cal_score(g);
		vector<int> y;
		bool optimal=true;
		double p,d;
		clock_t start_time=clock();
		int act_iter=0;
		if(_algo==0)
			optimal=decode(g,y,p,d, act_iter);
		else if(_algo==1)
			optimal=decode_naive_dd(g,y,p,d, act_iter);
		else
			decode_gurobi(g,y);
		double elapse = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
		timecost+=elapse;
		total_p+=p;
		total_d+=d;
		total_iter+=act_iter;

		for(int j=0;j<g.y.size();j++){
			ofs<<g.y[j]<<'\t'<<y[j]<<endl;
			if(y[j]!=g.y[j])
				err++;
		}
		total+=g.y.size();
		ofs<<endl;
		if(optimal)
			opts++;
	}
	ofs.close();
	//cout<<"opt: "<<opts<<'/'<<_graphs.size()<<"="<<double(opts)/_graphs.size()<<endl;
	//cout<<"primal: "<<total_p/_graphs.size()<<"\tdual: "<<total_d/_graphs.size()<<endl;
	cout<<dual_iter<<'\t'<<double(opts)/_graphs.size()<<'\t'<<total_p/_graphs.size()<<'\t'<<total_d/_graphs.size()<<"\t"<<timecost<<'\t'<<double(err)/total<<'\t'<<timecost/total_iter<<endl;
	return true;
}
bool mrf::load_model(char *model_file){
	ifstream ifs;
	ifs.open(model_file);
	if(!ifs.is_open()){
		cout<<"can not open model file: "<<model_file<<endl;
		return false;
	}
	char line[8192];
	ifs.getline(line,sizeof(line)-1);
	int lambda_sz=atoi(line);
	_lambda.resize(lambda_sz,0);
	for(int i=0;i<lambda_sz;i++){
		ifs.getline(line,sizeof(line)-1);
		_lambda[i]=atof(line);
	}
	ifs.close();
	return true;
}
void mrf::save_model(char *model_file){
	ofstream ofs;
	ofs.open(model_file);
	ofs<<_lambda.size()<<endl;
	for(int i=0;i<_lambda.size();i++)
		ofs<<_lambda[i]<<endl;
	ofs.close();
	ofs.clear();
}
int mrf::update(graph &g, vector<int> &y, int time){
	int err=0;
	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		bool is_key=true;
		bool is_y=true;
		for(int j=0;j<c.nodes.size();j++){
			if(y[c.nodes[j].first]!=c.nodes[j].second)
				is_y=false;
			if(g.y[c.nodes[j].first]!=c.nodes[j].second)
				is_key=false;
		}
		if(is_key)
			for(int j=0;j<c.fvector.size();j++){
				_lambda[c.fvector[j]]+=c.fvalue[j];
				_sum[c.fvector[j]]+=c.fvalue[j]*time;
			}

		if(is_y)
			for(int j=0;j<c.fvector.size();j++){
				_lambda[c.fvector[j]]-=c.fvalue[j];
				_sum[c.fvector[j]]-=c.fvalue[j]*time;
			}
	}
	for(int i=0;i<y.size();i++)
		if(y[i]!=g.y[i])
			err++;
	return err;
}
void mrf::cal_score(graph &g){
	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		c.score=0;
		for(int j=0;j<c.fvector.size();j++){
			c.score+=_lambda[c.fvector[j]]*c.fvalue[j];
		}
	}
}
bool mrf::decode_gurobi(graph &g, vector<int> &y){
	ofstream ofs;
	ofs.open("_lp.lp");
//	ofs.setf(ios::showpos);
	ofs<<"Maximize"<<endl;
	ofs<<"value: ";
	vector<string> constraints;
	vector<string> new_variables;
	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		if(c.score>=0)
			ofs<<" +"<<c.score;
		else
			ofs<<" "<<c.score;
		if(c.nodes.size()==1)
			ofs<<" y_"<<c.nodes[0].first<<"_"<<c.nodes[0].second;
		else{
			
			char tmp[1024*8];
			char tmp1[1024*8];
			for(int j=0;j<c.nodes.size();j++){
				sprintf(tmp,"t_%d - y_%d_%d <= 0",i,c.nodes[j].first,c.nodes[j].second);
				constraints.push_back(tmp);
			}
			sprintf(tmp,"t_%d", i);
			new_variables.push_back(tmp);
			for(int j=0;j<c.nodes.size();j++){
				sprintf(tmp1," - y_%d_%d",c.nodes[j].first,c.nodes[j].second);
				strcat(tmp,tmp1);
			}
			sprintf(tmp1," >= -%d", c.nodes.size()-1);
			strcat(tmp,tmp1);
			constraints.push_back(tmp);
			sprintf(tmp,"t_%d", i);
			
			ofs<<" t_"<<i;
		}
	}
	ofs<<endl;
	//constraints
	//bigram
	ofs<<"Subject To"<<endl;
	int constraint=0;
	for(int i=0;i<constraints.size();i++)
		ofs<<"C"<<constraint++<<": "<<constraints[i]<<endl;
	for(int i=0;i<g.ysz.size();i++){
		ofs<<"C"<<constraint++<<": ";
		for(int j=0;j<g.ysz[i];j++)
			ofs<<" + y_"<<i<<"_"<<j;
		ofs<<" = 1"<<endl;
	}
	
	ofs<<"Binary"<<endl;
	for(int i=0;i<g.ysz.size();i++){
		for(int j=0;j<g.ysz[i];j++)
			ofs<<"y_"<<i<<"_"<<j<<endl;
	}
	for(int i=0;i<new_variables.size();i++)
		ofs<<new_variables[i]<<endl;
	ofs<<"End"<<endl;
	ofs.close();
	ofs.clear();
	system("gurobi_cl ResultFile=lp.sol _lp.lp");
	ifstream ifs;
	ifs.open("lp.sol");
	char line[1024*8];
	y.clear();
	y.resize(g.y.size(),-1);
	ifs.getline(line,sizeof(line)-1);
	while(!ifs.eof()){
		ifs.getline(line,sizeof(line)-1);
		if(line[0]!='y')
			continue;
		char *p=strchr(line,' ');
		*p=0;
		p++;
		int x=atoi(p);
		if(x){
			vector<char *> v;
			split_string(line,"_",v);
			y[atoi(v[1])]=atoi(v[2]);
		}
	}
	ifs.close();ifs.clear();
	return true;
}
bool mrf::decode(graph &g, vector<int> &y, double &primal_obj, double &dual_obj, int &act_iter){
	bool is_optimal=false;
	vector<vector<double> > theta;
	theta.resize(g.y.size());//dual variable
	for(int i=0;i<g.y.size();i++)
		theta[i].resize(g.ysz[i],0);

	vector<vector<double> > old_theta=theta;
	vector<vector<double> > gtheta=theta;

	double old_obj=INF;


	double max_primal=-INF;
	//reparameterization
	vector<vector<vector<vector<double> > > > C_all(g.y.size());
	vector<vector<double> > C(g.y.size());
	for(int i=0;i<g.y.size();i++){
		C_all[i].resize(g.y.size());
		C[i].resize(g.y.size(),0);
		for(int j=i+1;j<g.y.size();j++){
			C_all[i][j].resize(g.ysz[i]);
			for(int k=0;k<g.ysz[i];k++){
				C_all[i][j][k].resize(g.ysz[j],0);
			}
		}
	}
			
	
	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		if(c.nodes.size()==1)
			continue;
		//change this if not chain
		if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
			continue;
		if(c.score<0){
			for(int j=0;j<c.nodes.size();j++){
				for(int k=j+1;k<c.nodes.size();k++){
					C_all[c.nodes[j].first][c.nodes[k].first][c.nodes[j].second][c.nodes[k].second]-=c.score;
				}
			}
		}
	}

	for(int i=0;i<g.y.size();i++)
		for(int j=i+1;j<g.y.size();j++)
			for(int k=0;k<g.ysz[i];k++)
				for(int l=0;l<g.ysz[j];l++)
					if(C[i][j]<C_all[i][j][k][l])
						C[i][j]=C_all[i][j][k][l];

	vector<double> B(g.y.size(),0);
	for(int i=0;i<C.size();i++){
		for(int j=i+1;j<C[i].size();j++){
			B[i]-=C[i][j]/2*g.ysz[j];
			B[j]-=C[i][j]/2*g.ysz[i];
		}
	}

	//index the node
	int node_id=0;
	vector<vector<int> > node2id(g.y.size());
	vector<pair<int,int> > id2node;
	for(int i=0;i<g.y.size();i++){
		node2id[i].resize(g.ysz[i],-1);
		for(int j=0;j<g.ysz[i];j++){
			id2node.push_back(pair<int,int>(i,j));
			node2id[i][j]=node_id++;
		}
	}
	

	vector<vector<int> > mono;
	vector<double> phi;
	//linear terms
	for(int i=0;i<g.y.size();i++){
		for(int j=0;j<g.ysz[i];j++){
			vector<int> mn(1);
			mn[0]=node2id[i][j];
			mono.push_back(mn);
			phi.push_back(-theta[i][j]);
		}
	}



	vector<vector<vector<vector<int> > > > C2mono(g.y.size());
	for(int i=0;i<g.y.size();i++){
		C2mono[i].resize(g.y.size());
		for(int j=i+1;j<g.y.size();j++){
			C2mono[i][j].resize(g.ysz[i]);
			for(int k=0;k<g.ysz[i];k++){
				C2mono[i][j][k].resize(g.ysz[j],-1);
				if(C[i][j]<0.000001)
					continue;
				for(int l=0;l<g.ysz[j];l++){
					C2mono[i][j][k][l]=mono.size();
					vector<int> mn(2);
					mn[0]=node2id[i][k];mn[1]=node2id[j][l];
					mono.push_back(mn);
					phi.push_back(C[i][j]);
				}
			}
		}
	}

	int vsz=node_id;
	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		if(c.nodes.size()==1)
			continue;
		if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
			continue;
		if(c.nodes.size()==2){
			int &c2m=C2mono[c.nodes[0].first][c.nodes[1].first][c.nodes[0].second][c.nodes[1].second];
			if(c2m==-1){
				c2m=mono.size();
				vector<int> mn(2);
				mn[0]=node2id[c.nodes[0].first][c.nodes[0].second];
				mn[1]=node2id[c.nodes[1].first][c.nodes[1].second];
				mono.push_back(mn);
				phi.push_back(c.score);
			}else{
				phi[c2m]+=c.score;
			}
		}else{
			if(c.score>0){
				for(int j=0;j<c.nodes.size();j++){
					vector<int> mn(2);
					mn[0]=node2id[c.nodes[j].first][c.nodes[j].second];
					mn[1]=vsz;
					mono.push_back(mn);
					phi.push_back(c.score);
				}
				vector<int> mn(1);
				mn[0]=vsz;
				mono.push_back(mn);
				phi.push_back(c.score*(1-int(c.nodes.size())));
				vsz++;
			}else if(c.score<0){
				for(int j=0;j<c.nodes.size();j++){
					for(int k=j+1;k<c.nodes.size();k++){
						int mono_id=C2mono[c.nodes[j].first][c.nodes[k].first][c.nodes[j].second][c.nodes[k].second];
						phi[mono_id]+=c.score;
					}
				}

				if(c.nodes.size()==3){
					//max y (x_1 + x_2 + x_3 - 1)
					for(int j=0;j<c.nodes.size();j++){
						vector<int> mn(2);
						mn[0]=node2id[c.nodes[j].first][c.nodes[j].second];
						mn[1]=vsz;
						mono.push_back(mn);
						phi.push_back(-c.score);
					}
					vector<int> mn(1);
					mn[0]=vsz;
					mono.push_back(mn);
					phi.push_back(c.score);
					vsz++;
				}else{
					for(int j=0;j<c.nodes.size();j++){
						vector<int> mn(2);
						mn[0]=node2id[c.nodes[j].first][c.nodes[j].second];
						mn[1]=vsz;
						mono.push_back(mn);
						phi.push_back(-2*c.score);
					}
					vector<int> y0(1);
					y0[0]=vsz;
					mono.push_back(y0);
					phi.push_back(3*c.score);
					vsz++;
					for(int a=1;a<=c.nodes.size()-4;a++){
						for(int j=0;j<c.nodes.size();j++){
							vector<int> mn(2);
							mn[0]=node2id[c.nodes[j].first][c.nodes[j].second];
							mn[1]=vsz;
							mono.push_back(mn);
							phi.push_back(-c.score);
						}
						vector<int> ya(1);
						ya[0]=vsz;
						mono.push_back(ya);
						phi.push_back((a+2)*c.score);
						vsz++;
					}
				}
			}
		}
	}



	vector<double> bs(vsz,0);
	for(int i=0;i<g.y.size();i++)
		for(int j=0;j<g.ysz[i];j++)
			bs[node2id[i][j]]+=B[i];

	vector<double> as;
	vector<int> is;
	vector<int> js;
	for(int i=0;i<mono.size();i++){
		if(mono[i].size()==1){
			bs[mono[i][0]]+=phi[i];
			continue;
		}
		is.push_back(mono[i][0]);
		js.push_back(mono[i][1]);
		as.push_back(phi[i]);
	}
	vector<double> old_bs=bs;
	double step=8;
	int iter;
	for(iter=0;iter<_dual_iter;iter++){
		for(int i=0;i<theta.size();i++)
			for(int j=0;j<theta[i].size();j++)
				theta[i][j]=old_theta[i][j]+gtheta[i][j]*step;

		double obj=0;
		//get chain
		//viterbi
		vector<vector<int> > best_prev(g.y.size());
		vector<double> prev_score;
		vector<double> cur_score;
		for(int i=0;i<g.ysz.size();i++)
			best_prev[i].resize(g.ysz[i],-1);
		
		
		for(int i=0;i<g.node2clique.size();i++){
			prev_score=cur_score;
			cur_score.clear();
			cur_score.resize(g.ysz[i]);
			for(int j=0;j<g.node2clique[i].size();j++){
				double s=theta[i][j];
				if(g.node2clique[i][j]!=-1)
					s+=g.cliques[g.node2clique[i][j]].score;
				if(i){
					for(int k=0;k<g.ysz[i-1];k++){
						double s1=prev_score[k]+s;
						if(g.edge2clique[i-1][i].size() && g.edge2clique[i-1][i][k][j]!=-1)
							s1+=g.cliques[g.edge2clique[i-1][i][k][j]].score;
						if(best_prev[i][j]==-1 || s1>cur_score[j]){
							cur_score[j]=s1;
							best_prev[i][j]=k;
						}
					}
				}else{
					cur_score[j]=s;
				}
			}
		}
		//back trace
		double maxs=0;
		vector<int> y1(g.y.size(),-1);
		for(int i=0;i<cur_score.size();i++){
			if(i==0 || cur_score[i] > maxs){
				maxs=cur_score[i];
				y1.back()=i;
			}
		}
		for(int i=y1.size()-2;i>=0;i--)
			y1[i]=best_prev[i+1][y1[i+1]];
		
		obj+=maxs;


		//graph cut
		//linear terms
		int nid=0;
		for(int i=0;i<g.y.size();i++){
			for(int j=0;j<g.ysz[i];j++){
				bs[nid]=old_bs[nid]-theta[i][j];
				nid++;
			}
		}

		

		int n=bs.size();
		int m=as.size();
		bool *z=new bool [n];
		double w=bqp(n,m,&is[0],&js[0],&as[0],&bs[0],z);
		obj+=w;

		vector<vector<int> > y2(g.y.size());
		for(int i=0;i<g.y.size();i++)
			y2[i].resize(g.ysz[i],-1);
		for(int i=0;i<g.cliques.size();i++){
			clique &c=g.cliques[i];
			if(c.nodes.size()==1)
				continue;
			if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
				continue;
			for(int j=0;j<c.nodes.size();j++)
				y2[c.nodes[j].first][c.nodes[j].second]=0;
		}

	

		for(int i=0;i<id2node.size();i++){
			if(z[i])
				y2[id2node[i].first][id2node[i].second]=1;
		}
		delete [] z;
		double primal=0;
		for(int i=0;i<C.size();i++){
			for(int j=i+1;j<C[i].size();j++){
				obj-=C[i][j];
			}
		}
		for(int i=0;i<B.size();i++)
			obj-=B[i];
		for(int i=0;i<g.cliques.size();i++){
			clique &c=g.cliques[i];
			bool selected=true;
			for(int j=0;j<c.nodes.size();j++){
				if(y1[c.nodes[j].first]!=c.nodes[j].second){
					selected=false;
					break;
				}
			}
			if(selected)
				primal+=c.score;
		}
		if(primal>max_primal){
			max_primal=primal;
			y=y1;
		}
		if(old_obj<obj){
			step*=0.5;
			continue;
		}
		

		old_obj=obj;
		old_theta=theta;
		bool agree=true;
		for(int i=0;i<g.y.size();i++){
			fill(gtheta[i].begin(),gtheta[i].end(),0);
			for(int j=0;j<g.ysz[i];j++){
				if(y2[i][j]==-1)
					continue;
				if(y1[i]==j)
					gtheta[i][j]--;
				if(y2[i][j]==1)
					gtheta[i][j]++;
				if(y1[i]==j && y2[i][j]!=1)
					agree=false;
				else if(y1[i]!=j && y2[i][j]==1)
					agree=false;
			}
		}
//		if(obj+1<max_primal)
//			cout<<"iter "<<iter<<"\tdual "<<obj<<"\tprimal "<<primal<<endl;
			
		//if(agree || max_primal+0.001>obj){//ner
		if(agree|| max_primal>obj){
			is_optimal=true;
			break;
		}
	}
	act_iter=iter;
	primal_obj=max_primal;
	dual_obj=old_obj;
	return is_optimal;
}
int mrf::read_data(char *data_file){
	ifstream ifs;
	ifs.open(data_file);
	if(!ifs.is_open()){
		cout<<"can not open file: "<<data_file<<endl;
		return false;
	}
	
	//format 
	//Y_0 YSZ_0 Y_1 YSZ_1 ...
	//NODE_ID_0 LABEL_ID_0 ... NODE_ID_N LABEL_ID_N
	//FEATURE_ID_0 FEATURE_VALUE_0 ...
	//NODE_ID_0 LABEL_ID_0 ... NODE_ID_N LABEL_ID_N
	//FEATURE_ID_0 FEATURE_VALUE_0 ...
	//...
	//<BLANK LINE>
	//NODE_NUM Y_0 Y_1 ...
	char *line=new char[MAXSTRLEN];
	int lambda_sz=0;
	vector<char *> v;
	while(!ifs.eof()){
		ifs.getline(line,MAXSTRLEN-1);
		if(line[0]==0)
			continue;

		split_string(line," ",v);

		graph g;
		for(int i=0;i<v.size();i+=2){
			g.y.push_back(atoi(v[i]));
			g.ysz.push_back(atoi(v[i+1]));
		}
		
		while(!ifs.eof()){
			ifs.getline(line,MAXSTRLEN-1);
			if(line[0]==0)
				break;
			split_string(line," ",v);
			int clique_sz=v.size()/2;
			clique c;
			for(int i=0;i<v.size();i+=2)
				c.nodes.push_back(pair<int,int>(atoi(v[i]),atoi(v[i+1])));
			//SORT

			ifs.getline(line,MAXSTRLEN-1);
			split_string(line," ",v);
			int fsz=v.size()/2;
			for(int i=0;i<v.size();i+=2){
				c.fvector.push_back(atoi(v[i]));
				c.fvalue.push_back(atof(v[i+1]));
			}
			for(int i=0;i<fsz;i++)
				if(lambda_sz<=c.fvector[i])
					lambda_sz=c.fvector[i]+1;

			g.cliques.push_back(c);
		}
		//build lattice, allocate pairwise cliques for high order cliques
		g.node2clique.resize(g.y.size());
		for(int i=0;i<g.y.size();i++)
			g.node2clique[i].resize(g.ysz[i],-1);

		for(int i=0;i<g.cliques.size();i++){
			vector<pair<int, int> > &ns=g.cliques[i].nodes;
			if(ns.size()==1){
				g.node2clique[ns[0].first][ns[0].second]=i;
			}
		}

		for(int i=0;i<g.node2clique.size();i++){
			for(int j=0;j<g.node2clique[i].size();j++){
				if(g.node2clique[i][j]==-1){
					g.node2clique[i][j]=g.cliques.size();
					clique c;
					c.nodes.push_back(pair<int,int>(i,j));
					g.cliques.push_back(c);
				}
			}
		}



		set<pair<int,int> > allocate_edges;
		for(int i=0;i<g.cliques.size();i++){
			vector<pair<int,int> > &ns=g.cliques[i].nodes;
			if(ns.size()>1){
				for(int j=0;j+1<ns.size();j++){
					for(int k=j+1;k<ns.size();k++)
						allocate_edges.insert(pair<int,int>(ns[j].first,ns[k].first));
				}
			}
		}
		g.edge2clique.resize(g.y.size());
		for(int i=0;i<g.y.size();i++)
			g.edge2clique[i].resize(g.y.size());


		for(set<pair<int,int> >::iterator it=allocate_edges.begin();it!=allocate_edges.end();it++){
			g.edge2clique[it->first][it->second].resize(g.ysz[it->first]);
			for(int i=0;i<g.ysz[it->first];i++){
				g.edge2clique[it->first][it->second][i].resize(g.ysz[it->second],-1);
			}
		}

		for(int i=0;i<g.cliques.size();i++){
			vector<pair<int,int> > &ns=g.cliques[i].nodes;
			if(ns.size()==2)
				g.edge2clique[ns[0].first][ns[1].first][ns[0].second][ns[1].second]=i;
		}
		for(int i=0;i<g.edge2clique.size();i++){
			for(int j=i+1;j<g.edge2clique[i].size();j++){
				for(int k=0;k<g.edge2clique[i][j].size();k++){
					for(int l=0;l<g.edge2clique[i][j][k].size();l++){
						if(g.edge2clique[i][j][k][l]==-1){
							g.edge2clique[i][j][k][l]=g.cliques.size();
							clique c;
							vector<pair<int,int> > &ns=c.nodes;
							ns.push_back(pair<int,int>(i,k));
							ns.push_back(pair<int,int>(j,l));
							g.cliques.push_back(c);
						}
					}
				}
			}
		}
		_graphs.push_back(g);
	}
	ifs.close();
	ifs.clear();
	delete [] line;
	return lambda_sz;
}

/*
void mrf::decode_naive_dd(graph &g, vector<int> &y){
	vector<vector<vector<double> > > theta;
	int n=0;
	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		if(c.nodes.size()==1)
			continue;
		if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
			continue;
		n++;
	}
	theta.resize(n+1);
	
	for(int i=0;i<theta.size();i++){
		theta[i].resize(g.y.size());
		for(int j=0;j<g.y.size();j++)
			theta[i][j].resize(g.ysz[j],0);
	}


	vector<vector<vector<double> > > old_theta=theta;
	vector<vector<vector<double> > > gtheta=theta;

	double old_obj=INF;
	double step=1.0;
	for(int iter=0;iter<_dual_iter;iter++){
		for(int i=0;i<theta.size();i++)
			for(int j=0;j<theta[i].size();j++)
				for(int k=0;k<theta[i][j].size();k++)
					theta[i][j][k]=old_theta[i][j][k]-gtheta[i][j][k]*step;
		double obj=0;
		//get chain
		//viterbi
		vector<vector<int> > best_prev(g.y.size());
		vector<double> prev_score;
		vector<double> cur_score;
		for(int i=0;i<g.ysz.size();i++)
			best_prev[i].resize(g.ysz[i],-1);
		for(int i=0;i<g.node2clique.size();i++){
			prev_score=cur_score;
			cur_score.clear();
			cur_score.resize(g.ysz[i]);
			for(int j=0;j<g.node2clique[i].size();j++){
				double s=theta[0][i][j];
				if(g.node2clique[i][j]!=-1)
					s+=g.cliques[g.node2clique[i][j]].score;
				if(i){
					for(int k=0;k<g.ysz[i-1];k++){
						double s1=prev_score[k]+s;
						if(g.edge2clique[i-1][i].size() && g.edge2clique[i-1][i][k][j]!=-1)
							s1+=g.cliques[g.edge2clique[i-1][i][k][j]].score;
						if(best_prev[i][j]==-1 || s1>cur_score[j]){
							cur_score[j]=s1;
							best_prev[i][j]=k;
						}
					}
				}else{
					cur_score[j]=s;
				}
			}
		}
		//back trace
		double maxs=0;
		vector<int> y1(g.y.size(),-1);
		for(int i=0;i<cur_score.size();i++){
			if(i==0 || cur_score[i] > maxs){
				maxs=cur_score[i];
				y1.back()=i;
			}
		}
		for(int i=y1.size()-2;i>=0;i--)
			y1[i]=best_prev[i+1][y1[i+1]];
		
		obj+=maxs;

		vector<vector<vector<int> > > y2(theta.size());
		for(int i=0;i<y2.size();i++){
			y2[i].resize(g.y.size());
			for(int j=0;j<y2[i].size();j++){
				if(i==0)
					y2[i][j].resize(g.ysz[j],0);
				else
					y2[i][j].resize(g.ysz[j],-1);
			}
		}
		for(int i=0;i<g.y.size();i++)
			y2[0][i][y1[i]]=1;

		n=1;
		for(int i=0;i<g.cliques.size();i++){
			clique &c=g.cliques[i];
			if(c.nodes.size()==1)
				continue;
			if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
				continue;
			double s=0;
			for(int j=0;j<c.nodes.size();j++){
				s+=theta[n][c.nodes[j].first][c.nodes[j].second]+c.score;
			}
			obj+=s;
			if(s>=0){
				for(int j=0;j<c.nodes.size();j++){
					y2[n][c.nodes[j].first][c.nodes[j].second]=1;
				}
			}else{
				for(int j=0;j<c.nodes.size();j++){
					y2[n][c.nodes[j].first][c.nodes[j].second]=0;
				}
			}
			n++;
		}
	

		if(old_obj<obj){
			step*=0.5;
			continue;
		}
		y=y1;
		old_obj=obj;
		old_theta=theta;
		bool agree=true;
		for(int i=0;i<gtheta.size();i++){
			for(int j=0;j<g.y.size();j++){
				fill(gtheta[i][j].begin(),gtheta[i][j].end(),0);
				if(i==0)
					continue;
				for(int k=0;k<g.ysz[j];k++){
					if(y2[0][j][k]!=y2[i][j][k] && y2[i][j][k]!=-1){
						agree=false;
						gtheta[0][j][k]+=2*y2[0][j][k]-1;
						gtheta[i][j][k]+=2*y2[i][j][k]-1;
					}
				}
			}
		}
		if(agree)
			break;
	}
}
*/
bool mrf::decode_naive_dd(graph &g, vector<int> &y, double &primal_obj, double &dual_obj, int &act_iter){
	bool is_optimal=false;
	vector<vector<vector<double> > > theta;
	int n=0;
	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		if(c.nodes.size()==1)
			continue;
		if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
			continue;
		n++;
	}
	theta.resize(n+1);
	theta[0].resize(g.y.size());
		for(int j=0;j<g.y.size();j++)
			theta[0][j].resize(g.ysz[j],0);
	n=1;

	for(int i=0;i<g.cliques.size();i++){
		clique &c=g.cliques[i];
		if(c.nodes.size()==1)
			continue;
		if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
			continue;
		theta[n].resize(c.nodes.size());
		for(int j=0;j<c.nodes.size();j++)
			theta[n][j].resize(g.ysz[c.nodes[j].first],0);
		n++;
	}

	vector<vector<vector<double> > > old_theta=theta;
	vector<vector<vector<double> > > gtheta=theta;

	double max_primal=-INF;
	double old_obj=INF;
	double step=8;
	int iter;
	for(iter=0;iter<_dual_iter;iter++){
		for(int i=0;i<theta.size();i++)
			for(int j=0;j<theta[i].size();j++)
				for(int k=0;k<theta[i][j].size();k++)
					theta[i][j][k]=old_theta[i][j][k]-gtheta[i][j][k]*step;
		double obj=0;
		//get chain
		//viterbi
		vector<vector<int> > best_prev(g.y.size());
		vector<double> prev_score;
		vector<double> cur_score;
		for(int i=0;i<g.ysz.size();i++)
			best_prev[i].resize(g.ysz[i],-1);
		for(int i=0;i<g.node2clique.size();i++){
			prev_score=cur_score;
			cur_score.clear();
			cur_score.resize(g.ysz[i]);
			for(int j=0;j<g.node2clique[i].size();j++){
				double s=theta[0][i][j];
				if(g.node2clique[i][j]!=-1)
					s+=g.cliques[g.node2clique[i][j]].score;
				if(i){
					for(int k=0;k<g.ysz[i-1];k++){
						double s1=prev_score[k]+s;
						if(g.edge2clique[i-1][i].size() && g.edge2clique[i-1][i][k][j]!=-1)
							s1+=g.cliques[g.edge2clique[i-1][i][k][j]].score;
						if(best_prev[i][j]==-1 || s1>cur_score[j]){
							cur_score[j]=s1;
							best_prev[i][j]=k;
						}
					}
				}else{
					cur_score[j]=s;
				}
			}
		}
		//back trace
		double maxs=0;
		vector<int> y1(g.y.size(),-1);
		for(int i=0;i<cur_score.size();i++){
			if(i==0 || cur_score[i] > maxs){
				maxs=cur_score[i];
				y1.back()=i;
			}
		}
		for(int i=y1.size()-2;i>=0;i--)
			y1[i]=best_prev[i+1][y1[i+1]];
		
		obj+=maxs;

		vector<vector<vector<int> > > y2(theta.size());
		for(int i=0;i<y2.size();i++){
			y2[i].resize(theta[i].size());
			for(int j=0;j<y2[i].size();j++){
				y2[i][j].resize(theta[i][j].size(),0);
			}
		}
		for(int i=0;i<g.y.size();i++)
			y2[0][i][y1[i]]=1;

		n=1;
		for(int i=0;i<g.cliques.size();i++){
			clique &c=g.cliques[i];
			if(c.nodes.size()==1)
				continue;
			if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
				continue;
			double s=c.score;
			for(int j=0;j<c.nodes.size();j++){
				s-=theta[n][j][c.nodes[j].second];
				//debug
				//s+=theta[n][j][c.nodes[j].second];
			}
			if(s>=0){
				for(int j=0;j<c.nodes.size();j++){
					y2[n][j][c.nodes[j].second]=1;
				}
				obj+=s;
			}else{
				for(int j=0;j<c.nodes.size();j++){
					y2[n][j][c.nodes[j].second]=0;
				}
			}
			n++;
		}
	
		double primal=0;
		for(int i=0;i<g.cliques.size();i++){
			clique &c=g.cliques[i];
			bool selected=true;
			for(int j=0;j<c.nodes.size();j++){
				if(y1[c.nodes[j].first]!=c.nodes[j].second){
					selected=false;
					break;
				}
			}
			if(selected)
				primal+=c.score;
		}
		if(primal>max_primal){
			max_primal=primal;
			y=y1;
		}
		if(old_obj<obj){
			step*=0.5;
			continue;
		}

		old_obj=obj;
		old_theta=theta;
		bool agree=true;
		for(int i=0;i<gtheta.size();i++){
			for(int j=0;j<gtheta[i].size();j++){
				fill(gtheta[i][j].begin(),gtheta[i][j].end(),0);
			}
		}
		n=1;
		for(int i=0;i<g.cliques.size();i++){
			clique &c=g.cliques[i];
			if(c.nodes.size()==1)
				continue;
			if(c.nodes.size()==2 && c.nodes[0].first+1==c.nodes[1].first)
				continue;

			for(int j=0;j<c.nodes.size();j++){
//				double diff=double(y2[0][c.nodes[j].first][c.nodes[j].second]-y2[n][j][c.nodes[j].second])/2;
//				gtheta[0][c.nodes[j].first][c.nodes[j].second]+=diff;
//				gtheta[n][j][c.nodes[j].second]-=diff;
				if(y2[n][j][c.nodes[j].second] != y2[0][c.nodes[j].first][c.nodes[j].second]){
					agree=false;
					gtheta[0][c.nodes[j].first][c.nodes[j].second]+=2*y2[0][c.nodes[j].first][c.nodes[j].second]-1;
					gtheta[n][j][c.nodes[j].second]+=2*y2[n][j][c.nodes[j].second]-1;
				}
			}
			n++;
		}
		if(agree ||  max_primal>=obj*0.99){
			is_optimal=true;
			break;
		}
	}
	act_iter=iter;
	primal_obj=max_primal;
	dual_obj=old_obj;
	return is_optimal;
}
