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

int main(int argc, char** argv) {
  if (argc == 1) {
    printf("./knnTest (datafile) (queryfile) (#reference) (#query) (#dimension)"
           "(k) (alg) (Actual #dimension)\n");
    return 0;
  }

  assert(argc == 9);

  char *datafile = argv[1];
  char *queryfile = argv[2];

  char *distfile = "KNNdist.bin";
  char *idxfile = "KNNidx.bin";


  long int N = atoi(argv[3]);
  long int Q = atoi(argv[4]);
  long int D = atoi(argv[5]);
  long int k = atoi(argv[6]);
  int alg = atoi(argv[7]);
  int actual_dimension = atoi(argv[8]);

  knntype *data, *queries, *KNNdist, *KNNidx, *dp;
  cudaHostAlloc((void**)&data, N*D*sizeof(knntype), cudaHostAllocWriteCombined);
  cudaHostAlloc((void**)&queries, Q*D*sizeof(knntype), cudaHostAllocWriteCombined);
  KNNdist = (knntype*)malloc(Q*k*sizeof(knntype));
  KNNidx = (knntype*)malloc(Q*k*sizeof(knntype));
  cudaHostAlloc((void**)&dp, N*sizeof(knntype), cudaHostAllocWriteCombined);


  knntype *finalDist = (knntype*)malloc(k*Q*sizeof(knntype));
  knntype *finalIdx = (knntype*)malloc(k*Q*sizeof(knntype));

  printf("files: %s %s\n", datafile, queryfile);

  // TODO(shengren): There are individual kernels for D=50,128,1024,2048. The
  // kernel for general D is buggy. 'data' is NxD. 'queries' is QxD. Our test
  // input files are DxN and DxQ with D=100. So here we need to transpose the
  // input matrices and expand them to D=128.
  knntype *buf;

  buf = (knntype *)malloc(actual_dimension * N * sizeof(knntype));
  load(buf, datafile, actual_dimension * N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      data[i * D + j] = (j < actual_dimension) ? buf[j * N + i] : (knntype)0;
    }
  }
  free(buf);

  buf = (knntype *)malloc(actual_dimension * Q * sizeof(knntype));
  load(buf, queryfile, actual_dimension * Q);
  for (int i = 0; i < Q; ++i) {
    for (int j = 0; j < D; ++j) {
      queries[i * D + j] = (j < actual_dimension) ? buf[j * Q + i] : (knntype)0;
    }
  }
  free(buf);

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
    exit(EXIT_FAILURE);

  knnsexecute(plan, data, queries, KNNdist, KNNidx);

  //save(KNNdist, distfile, k*Q);
  //save(KNNidx, idxfile, k*Q);

  // 'KNNdist' and 'KNNidx' are Q by k matrices. The distances in 'KNNdist' are
  // squared Euclidean distances.

  FILE *file_knn_dist = fopen("dist.txt", "w");
  for (int i = 0; i < Q; ++i) {
    for (int j = 0; j < k; ++j) {
      if (j > 0) fprintf(file_knn_dist, " ");
      fprintf(file_knn_dist, "%.5f", KNNdist[i * k + j]);
    }
    fprintf(file_knn_dist, "\n");
  }
  fclose(file_knn_dist);

  FILE *file_knn_idx = fopen("idx.txt", "w");
  for (int i = 0; i < Q; ++i) {
    for (int j = 0; j < k; ++j) {
      if (j > 0) fprintf(file_knn_idx, " ");
      fprintf(file_knn_idx, "%.0f", KNNidx[i * k + j]);
    }
    fprintf(file_knn_idx, "\n");
  }
  fclose(file_knn_idx);

  cudaFreeHost(data);
  cudaFreeHost(queries);
  cudaFreeHost(KNNdist);
  cudaFreeHost(KNNidx);
  cudaFreeHost(dp);
  free(finalDist);
  free(finalIdx);

}
