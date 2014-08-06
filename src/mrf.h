#ifndef MRF_H
#define MRF_H
#include <vector>
using namespace std;
struct clique{
	vector<pair<int,int> > nodes;
	vector<int> fvector;
	vector<double> fvalue;
	double score;
};
struct graph{
	vector<int> y;
	vector<int> ysz;
	vector<clique> cliques;
	vector<vector<int> > node2clique;
	vector<vector<vector<vector<int> > > > edge2clique;//clique[edge2clique[i][j][y_i][y_j]]
};
class mrf{

public:
	mrf();
	~mrf();
	bool learn(char *train_file, char *model_file, int algo, int dual_iter);
	bool test(char *model_file, char *test_file, char *result_file,int algo, int dual_iter);
	bool load_model(char *model_file);
	void save_model(char *model_file);
	int update(graph &g, vector<int> &y, int time);
	void cal_score(graph &g);
	bool decode(graph &g, vector<int> &y,double &prim, double &dual, int &act_iter);
	bool decode_naive_dd(graph &g, vector<int> &y,double &prim, double &dual, int &act_iter);
	bool decode_gurobi(graph &g, vector<int> &y);
	int read_data(char *data_file);
	vector<graph> _graphs;
	vector<double> _lambda;
	vector<double> _sum;
	int _max_iter;
	int _dual_iter;
	int _algo;
};
#endif
