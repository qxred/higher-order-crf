
#ifndef BQP_H
#define BQP_H
double bqp(int n, int m, int *is, int *js, double *a, double *b, bool *x);
/*pseudo boolean quadratic programming:
	max \sum_{ij} a_{ij} x_ix_j + \sum_i b_i x_i
	s.t. x_i \in {0,1}
	where a_{ij}<=0;
*/
#endif
