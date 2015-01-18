#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "../include/cuknns.h"


void rand_init(knntype *a, int N){

  srand(time(NULL));
  for(int i=0; i<N; i++){
    a[i] = 100 * (knntype)rand() / RAND_MAX - 50;
  }

}

void load(knntype *a, char *file, int N){

  FILE* infile;
  size_t res;

  printf("Loading data from file: %s\n", file);

  if((infile=fopen(file, "rb"))==NULL){
    printf("Can't open input file\n");
  }

  res = fread(a, sizeof(knntype), N, infile);

  fclose(infile);

}

void save(knntype *a, char *file, int N){

  FILE* outfile;

  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb"))==NULL){
    printf("Can't open output file");
  }

  fwrite(a, sizeof(knntype), N, outfile);

  fclose(outfile);

}

void serial_dot(knntype *dot, knntype *data, int N, int D){

  for(int i=0; i<N; i++){
    knntype tmp= 0;
    for(int j=0; j<D; j++){
      knntype tt = data[i*D + j];
      tmp += tt * tt;
    }
    dot[i] = tmp;
  }

}

int main(int argc, char** argv){
  assert(argc == 8);

  char *datafile = argv[1];
  char *queryfile = argv[2];

  char *distfile = "KNNdist.bin";
  char *idxfile = "KNNidx.bin";


  long int N = atoi(argv[3]);
  long int Q = atoi(argv[4]);
  long int D = atoi(argv[5]);
  long int k = atoi(argv[6]);
  int alg = atoi(argv[7]);

  knntype *data, *queries, *KNNdist, *KNNidx, *dp;
  cudaHostAlloc((void**)&data, N*D*sizeof(knntype), cudaHostAllocWriteCombined);
  cudaHostAlloc((void**)&queries, Q*D*sizeof(knntype), cudaHostAllocWriteCombined);
  KNNdist = (knntype*)malloc(Q*k*sizeof(knntype));
  KNNidx = (knntype*)malloc(Q*k*sizeof(knntype));
  cudaHostAlloc((void**)&dp, N*sizeof(knntype), cudaHostAllocWriteCombined);


  knntype *finalDist = (knntype*)malloc(k*Q*sizeof(knntype));
  knntype *finalIdx = (knntype*)malloc(k*Q*sizeof(knntype));

  printf("files: %s %s\n", datafile, queryfile);

  load(data, datafile, N*D);
  load(queries, queryfile, Q*D);

  knnplan plan;

  //knnsplan(&plan, N, Q, D, k);

  plan.objects = N;
  plan.numQueries = Q;
  plan.dimentions = D;
  plan.k = k;
  plan.numStreams = 1;
  if (alg == 0)
    plan.pt2Function = &gpuknnsBitonic;
  else if (alg == 1)
    plan.pt2Function = &gpuknnsHeap;
  else
    exit(-1);

  knnsexecute(plan, data, queries, KNNdist, KNNidx);

  save(KNNdist, distfile, k*Q);
  save(KNNidx, idxfile, k*Q);

  cudaFreeHost(data);
  cudaFreeHost(queries);
  cudaFreeHost(KNNdist);
  cudaFreeHost(KNNidx);
  cudaFreeHost(dp);
  free(finalDist);
  free(finalIdx);

}
