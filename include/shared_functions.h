
#ifndef __DOUBLE__
typedef float knntype;
#else
typedef double knntype;
#endif

#ifndef CUKNN

typedef struct{
  double dst_time;
  double srch_time;
  double knn_time;
}knntimes;

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
  distfunction distF;
  dotKernel dotP;
}distFunctParam;

typedef struct{
  knntype *trg;
  knntype *src;
  int N;
  int Q;
}thread_arg;

#endif

__global__ void crpdot(knntype *B, knntype *A, int objects, int attributes);

__global__ void crpdot2_gen(knntype *B, knntype *A, int objects, int attributes, int P);

__global__ void crpdot2_50(knntype *B, knntype *A, int objects, int attributes, int P);

__global__ void crpdot2_128(knntype *B, knntype *A, int objects, int attributes, int P);

__global__ void crpdot2_1024(knntype *B, knntype *A, int objects, int attributes, int P);

__global__ void crpdot2_2048(knntype *B, knntype *A, int objects, int attributes, int P);

__global__ void dot(knntype *B, knntype *A, int objects, int attributes, int numQueries);


__global__ void pdot(knntype *dst, knntype *src, int objects, int numQueries);


__global__ void dot2(knntype *B, knntype *A, int objects, int attributes, int numQueries);


__global__ void pdot_v0(knntype *dst, knntype *src, int objects, int numQueries);


__global__ void pdot(knntype *dst, knntype *src, int objects, int numQueries, int iter);


__global__ void pdot_v2(knntype *dst, knntype *src, int objects, int numQueries, int iter);


__global__ void dot3(knntype *dst, knntype *src);


__device__ void warpReduce(volatile knntype *sX, int tid);

__global__ void dot4_gen(knntype *dst, knntype *src, int attributes);

__global__ void dot4_50(knntype *dst, knntype *src, int attributes);

__global__ void dot4(knntype *dst, knntype *src, int attributes);


void order_output(knntype *trg, knntype *src, int numQueries, int k, int numBlocks, int maxQueries);

__global__ void gpu_order_output(knntype *trg, knntype *src, int numQueries, int k, int numBlocks, int maxQueries);

__global__ void transposeNoBankConflicts(knntype *odata, knntype *idata, int width, int height, int nreps);


void transpose_naive(knntype *trg, knntype *src, int N, int Q);

void* ptranspose(void* argt);

__global__ void dot4_1024(knntype *dst, knntype *src, int attributes);


__global__ void dot4_2048(knntype *dst, knntype *src, int attributes);


void pdist_Q(knntype *dist, knntype *data, knntype *query, knntype *dotp, int objects, int attributes, int numQueries, cublasHandle_t handle,CUstream str, dotKernel *dotPKernel);

void pdist_N(knntype *dist, knntype *data, knntype *query, knntype *dotp, int objects, int attributes, int numQueries, cublasHandle_t handle, CUstream str);

void pdist_NT(knntype *dist, knntype *data, knntype *query, knntype *dotp, int objects, int attributes,int numQueries, cublasHandle_t handle, CUstream str, dotKernel *dotPKernel);
