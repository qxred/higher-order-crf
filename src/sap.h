#ifndef SAP_H
#define SAP_H
double sap(int n, int m, int *from, int *to, double *cap1, double *cap2, bool *cut);
/*
n: input, node size, 0 is the source, n-1 is the target
m: input, number of edge pairs ( i->j , j->i ), sizeof array from, to ,cap,cap2
from: input, source node of each edge
to: input, target node of each edge
cap1: input, cap1[k] is the capacity of the k^th edge i->j
cap2: input, cap2[k] is the capacity of the k^th edge's oppsite directed edge j->i
cut: output, cut[i] denotes whether i^th node is connected with source , space of cut should be allocated before calling this function.
*/
#endif
