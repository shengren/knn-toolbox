
#ifndef CUKNN
#define CUKNN

#include <cublas_v2.h>
#include "cuda.h"

#ifndef __DOUBLE__
typedef float knntype;
#else
typedef double knntype;
#endif

#define MEMTEST_ONLY 10
#define EXACT_KNN 5
#define FORCE_SINGLE_STREAM -5

typedef double (*knnptr2function)(knntype*, knntype*, knntype*, knntype*, int, int, int, int, int);


typedef void (*dotQfunction)(knntype*, knntype*, int);
typedef void (*dotCrpfunction)(knntype*, knntype*, int, int, int);

typedef struct{
  dotQfunction dotQ;
  int nthreadsQ;
  dotCrpfunction dotCrp;
  int nthreadsCrp;
  int externShared;
}dotKernel;

typedef void (*distfunction)(knntype*, knntype*, knntype*, knntype*, int, int, int, cublasHandle_t, CUstream, dotKernel*);

typedef struct{
  knnptr2function pt2Function;
  int objects;
  int numQueries;
  int dimentions;
  int k;
  int numStreams;
}knnplan;

typedef struct{
  double dst_time;
  double srch_time;
  double knn_time;
}knntimes;

typedef struct{
  distfunction distF;
  dotKernel dotP;
}distFunctParam;

typedef struct{
  knntype *trg;
  knntype *src;
  int N;
  int Q;
}thread_arg;


extern "C" void cuknnsHeap(knntype *dist, knntype *data, knntype *query, knntype *heap, knntype *index, knntype* dotp, int objects, int attributes, int numQueries, int qk, knntype *dotB, cublasHandle_t handle, CUstream stream, knntimes *times, int streamId, distFunctParam *distFunc);

extern "C" void cuknnsBarrientos(knntype *dist, knntype *data, knntype *query, knntype *heap, knntype *index, knntype* dotp, int objects, int attributes, int numQueries, int qk, knntype *dotB, cublasHandle_t handle, CUstream stream, knntimes *times, int streamId, distFunctParam *distFunc);

extern "C" void cuknnsBitonic(knntype *dist, knntype *data, knntype *query, knntype *index, knntype* dotp, knntype *d_dotB, knntype *distbuff, knntype *idxbuff, int objects, int attributes, int numQueries, int k, cublasHandle_t handle, CUstream stream, knntimes* times, int strId);

extern "C" void cuknnsBitonicSTR(knntype *dist, knntype *data, knntype *query, knntype *index, knntype* dotp, knntype *d_dotB, knntype *distbuff, knntype *idxbuff, int objects, int attributes, int numQueries, int k, cublasHandle_t handle, CUstream str, knntimes* times, int strId, distFunctParam *distFunc);

extern "C" void cuknnsSort(knntype *KNNdist, knntype *KNNidx,knntype *dist, knntype *data, knntype *query, knntype *index, knntype *dotp, int objects, int attributes, int numQueries, int k, cublasHandle_t handle, CUstream str, knntimes* times);


extern "C" void mergeStreams(knntype *dist, knntype *index, int k, int qk, int numStreams, CUstream str);

extern "C" double gpuknnsBitonic(knntype *query, knntype *data, knntype *values, knntype *indices, int objects, int numQueries, int attributes, int k, int numStreams);

extern "C" double gpuknnsLshBitonic(knntype *query, knntype *data, knntype *values, knntype *indices, knntype *dp, int objects, int numQueries, int attributes, int k, int numStreams);

extern "C" float gpuknnLSH(knntype *query, knntype *data, knntype *values, knntype *indices, knntype *dp, int objects, int numQueries, int attributes, int k, int numStreams, int *bucketSize,  int *query_offsets, int c, int *query_sizes);

extern "C" double gpuknnsBitonicMemTest(knntype *query, knntype *data, knntype *values, knntype *indices, knntype *dp, int objects, int numQueries, int attributes, int k, int numStreams);

extern "C" double gpuknnsHeap(knntype *query, knntype *data, knntype *values, knntype *indices, int objects, int numQueries, int attributes, int k, int numStreams);

extern "C" void knnsplan(knnplan *plan,long  int N, long  int Q, long  int D, long int k);

extern "C" void knnsexecute(knnplan plan, knntype *data, knntype *queries, knntype *KNNdist, knntype *KNNidx);


#endif

/*******************************/
