#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cublas_v2.h"
#include "../include/cuknns.h"
#include <time.h>
#include <sys/time.h>
#include <float.h>

#define ITERATIONS 10
#define ALG 2
#define NSTREAMS 4

int streams[NSTREAMS] = {1, 2, 4, 8};

void init(knntype *a, int N){

  srand(time(NULL));
  for(int i=0; i<N; i++){
    a[i] = 100 * (knntype)rand() / RAND_MAX - 50;
  }

}

extern "C" void knnsplan(knnplan *plan, long int N,long  int Q, long int D,long  int k){

  double times[NSTREAMS][ALG];
  double best_time[ALG];
  int best_stream[ALG];

  knntype *data, *queries, *KNNdist, *KNNidx, *dp;

  cudaHostAlloc((void**)&data, N*D*sizeof(knntype), cudaHostAllocWriteCombined);
  cudaHostAlloc((void**)&queries, Q*D*sizeof(knntype), cudaHostAllocWriteCombined);
  KNNdist = (knntype*)malloc(Q*k*sizeof(knntype));
  KNNidx = (knntype*)malloc(Q*k*sizeof(knntype));
  

  init(data, N*D);
  init(queries, Q*D);
  

  for(int s=0; s<NSTREAMS; s++){    

  double time1tmp = 0;
  double time2tmp = 0;
  for(int i=0; i<ITERATIONS; i++){

    time1tmp += gpuknnsBitonic(queries, data, KNNdist, KNNidx, N, Q, D, k, streams[s]);  

    time2tmp += gpuknnsHeap(queries, data, KNNdist, KNNidx, N, Q, D, k, streams[s]);
  }

  times[s][0] = time1tmp / ITERATIONS;
  times[s][1] = time2tmp / ITERATIONS;

  }


  for(int i=0; i<ALG; i++){
    best_time[i] = FLT_MAX;
    best_stream[i] = 0;

    for(int j=0; j<NSTREAMS; j++){

      if(best_time[i]>times[j][i]){
	best_time[i] = times[j][i];
	best_stream[i] = j;
      }
    }
  }

  int pic;

  if(best_time[0] < best_time[1]){
    pic = 0;
  }
  else{
    pic = 1;
  }

  
  switch (pic){

  case 0:
    plan->pt2Function = &gpuknnsBitonic;
    plan->numStreams = streams[best_stream[pic]];
    break;
  case 1:
    plan->pt2Function = &gpuknnsHeap;
    plan->numStreams = streams[best_stream[pic]];
    break;
  }

  plan->objects = N;
  plan->dimentions = D;
  plan->numQueries = Q;
  plan->k = k;

  cudaFreeHost(data);
  cudaFreeHost(queries);
  cudaFreeHost(KNNdist);
  cudaFreeHost(KNNidx);
  cudaFreeHost(dp);
}
