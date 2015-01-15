#include <stdio.h>
#include <stdlib.h>
#include "../include/cuknns.h"


extern "C" void knnsexecute(knnplan plan, knntype *data, knntype *queries, knntype *KNNdist, knntype *KNNidx){

  int N = plan.objects;
  int Q = plan.numQueries;
  int D = plan.dimentions;
  int k = plan.k;
  int numStreams = plan.numStreams;

  double time = plan.pt2Function(queries, data, KNNdist, KNNidx, N, Q, D, k, numStreams);

  printf("Time Elapsed: %f\n", time);

}
