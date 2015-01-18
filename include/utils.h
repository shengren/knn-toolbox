
#ifndef KNN_INC
#define KNN_INC

#define BLOCKSIZE_D 256
#define DIMENSIONS 128
#define TILE_DIM 16
#define BLOCK_ROWS 16
#define MAXBLOCKS 32768
#ifndef TUNE
#define BLOCKSIZE 256
#endif

#ifdef __DOUBLE__
typedef double knntype;
#else
typedef float knntype;
#endif

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

__device__ void warpReduce(volatile knntype *sX, int tid);
__device__ void halphwarpReduce(volatile knntype *sX, int tid);
__global__ void crpdot(knntype *B, knntype *A, int objects, int attributes){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  knntype sum = 0;
  knntype tmp;

  if(tid<objects){

#pragma unroll 8
    for(int i=0; i<attributes; i++){
      tmp = A[i*objects + tid];
      sum += tmp * tmp;
    }

    B[tid] = sum;
  }
}

__global__ void crpdot2_gen(knntype *B, knntype *A, int objects, int attributes, int P){

  int tid = threadIdx.x;

  extern __shared__ knntype ssX[];
  //__shared__ knntype ssX[64];
  //int dif = dim - blockDim.x;

  for(int i=0; i<P; i++){

    int point = blockIdx.x * P + i;

    if(point<objects){

      knntype* src = A + point * attributes + threadIdx.x;
      knntype* dst = B + point;
      knntype tmp = (tid<attributes) ? src[0] : 0;

      ssX[tid] = tmp * tmp;
      __syncthreads();

      if(blockDim.x>32){

	for(int ii=(blockDim.x>>1); ii>32; ii>>=1){
	  if(tid<ii) { ssX[tid] += ssX[tid + ii];}
	  __syncthreads();
	}


	if(tid<32){
	  warpReduce(ssX, tid);
	}

	__syncthreads();
      }
      else{
	if(tid<16){ halphwarpReduce(ssX, tid);}
	__syncthreads();
      }

      if(tid==0){
        dst[tid] = ssX[tid];
      }
    }

  }

}


__global__ void crpdot2_50(knntype *B, knntype *A, int objects, int attributes, int P){

  int tid = threadIdx.x;

  __shared__ knntype sX[64];
  int diff = 50 - 32;

  for(int i=0; i<P; i++){

    int point = blockIdx.x * P + i;

    if(point<objects){

      knntype* src = A + point * 50 + threadIdx.x;
      knntype* dst = B + point;
      knntype tmp = (tid<50) ? src[0] : 0;

      sX[tid] = tmp * tmp;
      __syncthreads();

      if(tid<32){
        warpReduce(sX, tid);
      }

      __syncthreads();

      if(tid==0){
        dst[tid] = sX[tid];
      }
    }
  }

}

__global__ void crpdot2_128(knntype *B, knntype *A, int objects, int attributes, int P){

  int tid = threadIdx.x;

  __shared__ knntype sX[DIMENSIONS];

  for(int i=0; i<P; i++){

    int point = blockIdx.x * P + i;

    if(point<objects){

      knntype* src = A + point*DIMENSIONS + threadIdx.x;
      knntype* dst = B + point;
      knntype tmp = src[0];
      sX[tid] = tmp * tmp;

      __syncthreads();

      if(tid<64) { sX[tid] += sX[tid + 64];}
      __syncthreads();
      if(tid<32){
	warpReduce(sX, tid);
      }

      __syncthreads();

      if(tid==0){
	dst[tid] = sX[tid];
      }
    }
  }

}


__global__ void crpdot2_1024(knntype *B, knntype *A, int objects, int attributes, int P){

  int tid = threadIdx.x;

  __shared__ knntype sX[1024];

  for(int i=0; i<P; i++){

    int point = blockIdx.x * P + i;

    if(point<objects){

      knntype* src = A + point*1024 + threadIdx.x;
      knntype* dst = B + point;
      knntype tmp = src[0];
      sX[tid] = tmp * tmp;
      tmp = src[512];
      sX[tid+512] = tmp * tmp;

      __syncthreads();

      if(tid<512){sX[tid] += sX[tid + 512];}
      __syncthreads();
      if(tid<256) {sX[tid] += sX[tid + 256];}
      __syncthreads();
      if(tid<128) {sX[tid] += sX[tid + 128];}
      __syncthreads();
      if(tid<64) { sX[tid] += sX[tid + 64];}
      __syncthreads();
      if(tid<32){
	warpReduce(sX, tid);
      }

      __syncthreads();

      if(tid==0){
        dst[tid] = sX[tid];
      }
    }
  }

}


#if defined(CUARCH) && (CUARCH>=20)
__global__ void crpdot2_2048(knntype *B, knntype *A, int objects, int attributes, int P){

  int tid = threadIdx.x;

  __shared__ knntype sX[2048];

  for(int i=0; i<P; i++){

    int point = blockIdx.x * P + i;

    if(point<objects){

      knntype* src = A + point*2048 + threadIdx.x;
      knntype* dst = B + point;
      knntype tmp = src[0];
      sX[tid] = tmp * tmp;
      tmp = src[512];
      sX[tid+512] = tmp * tmp;
      tmp = src[1024];
      sX[tid+1024] = tmp*tmp;
      tmp = src[1024+512];
      sX[tid+512+1024] = tmp*tmp;

      __syncthreads();

      sX[tid] += sX[tid + 1024];
      sX[tid+512] += sX[tid + 512 + 1024];
      __syncthreads();
      if(tid<512){sX[tid] += sX[tid + 512];}
      __syncthreads();
      if(tid<256) {sX[tid] += sX[tid + 256];}
      __syncthreads();
      if(tid<128) {sX[tid] += sX[tid + 128];}
      __syncthreads();
      if(tid<64) { sX[tid] += sX[tid + 64];}
      __syncthreads();
      if(tid<32){
	warpReduce(sX, tid);
      }

      __syncthreads();

       if(tid==0){
        dst[tid] = sX[tid];
      }
    }
  }

}
#endif

#if defined(CUARCH) && (CUARCH<20)
__global__ void crpdot2_2048(knntype *B, knntype *A, int objects, int attributes, int P){

  int tid = threadIdx.x;

  __shared__ knntype sX[1024];
  knntype res = 0;

  for(int i=0; i<P; i++){

    int point = blockIdx.x * P + i;

    if(point<objects){

      knntype* src = A + point*2048 + threadIdx.x;
      knntype* dst = B + point;
      knntype tmp = src[0];
      sX[tid] = tmp * tmp;
      tmp = src[512];
      sX[tid+512] = tmp * tmp;
      tmp = src[1024];


      __syncthreads();

      __syncthreads();
      if(tid<512){sX[tid] += sX[tid + 512];}
      __syncthreads();
      if(tid<256) {sX[tid] += sX[tid + 256];}
      __syncthreads();
      if(tid<128) {sX[tid] += sX[tid + 128];}
      __syncthreads();
      if(tid<64) { sX[tid] += sX[tid + 64];}
      __syncthreads();
      if(tid<32){
	warpReduce(sX, tid);
      }

      __syncthreads();


      res = sX[tid];

      sX[tid] = tmp*tmp;
      tmp = src[1024+512];
      sX[tid+512] = tmp*tmp;


      __syncthreads();
      if(tid<512){sX[tid] += sX[tid + 512];}
      __syncthreads();
      if(tid<256) {sX[tid] += sX[tid + 256];}
      __syncthreads();
      if(tid<128) {sX[tid] += sX[tid + 128];}
      __syncthreads();
      if(tid<64) { sX[tid] += sX[tid + 64];}
      __syncthreads();
      if(tid<32){
	warpReduce(sX, tid);
      }

      __syncthreads();

       if(tid==0){
        dst[tid] = res + sX[tid];
      }
    }
  }

}
#endif

__global__ void dot(knntype *B, knntype *A, int objects, int attributes, int numQueries){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  knntype sum = 0;
  knntype tmp;

  if(tid<objects){

#pragma unroll 8
    for(int i=0; i<attributes; i++){
      tmp = A[i*objects + tid];
      sum += tmp * tmp;
    }

#pragma unroll 8
    for(int i=0; i<numQueries; i++){
      B[i*objects + tid] += sum;
    }

  }
}


__global__ void pdot(knntype *dst, knntype *src, int objects, int numQueries){

  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  if(tid<objects){

    knntype tmp = src[tid];

#pragma unroll 8
    for(int i=0; i<numQueries; i++){
      dst[i * objects + tid] += tmp;
    }
  }

}

__global__ void dot2(knntype *B, knntype *A, int objects, int attributes, int numQueries){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  knntype sum = 0;
  knntype tmp;

  if(tid<objects){

#pragma unroll 8
    for(int i=0; i<attributes; i++){
      tmp = A[i*numQueries + tid];
      sum += tmp * tmp;
    }

#pragma unroll 8
    for(int i=0; i<numQueries; i++){
      B[i*objects + tid] = sum;
    }

  }
}

/********************************************************************/


__global__ void pdot_v0(knntype *dst, knntype *src, int objects, int numQueries){

  int tidx = threadIdx.x;
  int blockx = blockIdx.x;
  int blocky = blockIdx.y;

  int idx = blockx * BLOCKSIZE_D + tidx;

  if(idx<numQueries){

    knntype tmp = src[idx];

    knntype *tmp_dst = dst + blocky * numQueries + blockx * blockDim.x + tidx;

    tmp_dst[0] += tmp;
  }
}

__global__ void pdot(knntype *dst, knntype *src, int objects, int numQueries, int iter){

  int tidx = threadIdx.x;
  int tidystr = blockIdx.x * iter * blockDim.y;

#pragma unroll 2
  for(int i=0; i<iter; i++){

    int tidy = tidystr + i * blockDim.y + threadIdx.y;

    if(tidy<objects){
      knntype tmp = src[tidy];
      knntype *tmp_dst = dst + tidy * numQueries + tidx;

      tmp_dst[0] += tmp;
    }

  }

}

__global__ void pdot_v2(knntype *dst, knntype *src, int objects, int numQueries, int iter){

  int tidx = threadIdx.x;
  int tidystr = blockIdx.x * iter;
  int qidx = blockIdx.y * BLOCKSIZE_D + tidx;

  if(qidx<numQueries){

    for(int i=0; i<iter; i++){

      int tidy = tidystr + i;


      knntype tmp = src[tidy];
      knntype *tmp_dst = dst + tidy * numQueries + blockIdx.y * BLOCKSIZE_D + tidx;

      tmp_dst[0] += tmp;
    }
  }

}

__global__ void dot3(knntype *dst, knntype *src){

  knntype dot_val = src[blockIdx.x];
  dst += blockIdx.x * blockDim.x + threadIdx.x;

  dst[0] += dot_val;
}

__device__ void warpReduce(volatile knntype *sX, int tid){

  sX[tid] += sX[tid + 32];
  sX[tid] += sX[tid + 16];
  sX[tid] += sX[tid + 8];
  sX[tid] += sX[tid + 4];
  sX[tid] += sX[tid + 2];
  sX[tid] += sX[tid + 1];

}

__device__ void halphwarpReduce(volatile knntype *sX, int tid){

  sX[tid] += sX[tid + 16];
  sX[tid] += sX[tid + 8];
  sX[tid] += sX[tid + 4];
  sX[tid] += sX[tid + 2];
  sX[tid] += sX[tid + 1];

}



__global__ void dot4_gen(knntype *dst, knntype *src, int attributes){

  extern __shared__ knntype dX[];

  int tid = threadIdx.x;
  int block = blockIdx.x;

  int offset = block * attributes + tid;
  //int diff = dim - blockDim.x;

  src += offset;

  knntype tmp = (tid<attributes) ? src[0] : 0;
  dX[tid] = tmp * tmp;
  __syncthreads();


  if(blockDim.x>32){

    for(int ii=blockDim.x>>1; ii>32; ii>>=1){
      if(tid<ii) { dX[tid] += dX[tid + ii];}
      __syncthreads();
    }

    if(tid<32){
      warpReduce(dX, tid);
    }
    __syncthreads();

  }
  else{
    if(tid<16){ halphwarpReduce(dX, tid);}
    __syncthreads();
  }

  if(tid==0){
    dst[block] = dX[tid];
  }



}

__global__ void dot4_50(knntype *dst, knntype *src, int attributes){

  __shared__ knntype sX[64];

  int tid = threadIdx.x;
  int block = blockIdx.x;

  int offset = block * 50 + tid;
  int diff = 50-32;

  src += offset;

  knntype tmp = (tid<50) ? src[0] : 0;
  sX[tid] = tmp * tmp;
  __syncthreads();

  if(tid<32){
    warpReduce(sX, tid);
  }

  __syncthreads();

  if(tid==0){
    dst[block] = sX[tid];
  }


}

__global__ void dot4(knntype *dst, knntype *src, int attributes){

  __shared__ knntype sX[DIMENSIONS];

  int tid = threadIdx.x;
  int block = blockIdx.x;

  int offset = block * DIMENSIONS + tid;

  src += offset;

  knntype tmp = src[0];
  sX[tid] = tmp * tmp;

  __syncthreads();

  if(tid<64) { sX[tid] += sX[tid + 64];}
  __syncthreads();
  if(tid<32){
    warpReduce(sX, tid);
  }

  __syncthreads();

  if(tid==0){
    dst[block] = sX[tid];
  }

}

__global__ void dot4_1024(knntype *dst, knntype *src, int attributes){

  __shared__ knntype sX[1024];

  int tid = threadIdx.x;
  int block = blockIdx.x;

  int offset = block * 1024 + tid;

  src += offset;

  knntype tmp = src[0];
  sX[tid] = tmp * tmp;
  tmp = src[512];
  sX[tid+512] = tmp*tmp;

  __syncthreads();

  if(tid<512){sX[tid] += sX[tid + 512];}
  __syncthreads();
  if(tid<256) {sX[tid] += sX[tid + 256];}
  __syncthreads();
  if(tid<128) {sX[tid] += sX[tid + 128];}
  __syncthreads();
  if(tid<64) { sX[tid] += sX[tid + 64];}
  __syncthreads();
  if(tid<32){
    warpReduce(sX, tid);
  }

  __syncthreads();

  if(tid==0){
    dst[block] = sX[tid];
  }

}

#if defined(CUARCH) && (CUARCH>=20)
__global__ void dot4_2048(knntype *dst, knntype *src, int attributes){

  __shared__ knntype sX[2048];

  int tid = threadIdx.x;
  int block = blockIdx.x;

  int offset = block * 2048 + tid;

  src += offset;

  knntype tmp = src[0];
  sX[tid] = tmp * tmp;
  tmp = src[512];
  sX[tid+512] = tmp*tmp;
  tmp = src[1024];
  sX[tid+1024] = tmp*tmp;
  tmp = src[1024+512];
  sX[tid+512+1024] = tmp*tmp;

  __syncthreads();

  sX[tid] += sX[tid + 1024];
  sX[tid+512] += sX[tid + 512 + 1024];
  __syncthreads();
  if(tid<512){sX[tid] += sX[tid + 512];}
  __syncthreads();
  if(tid<256) {sX[tid] += sX[tid + 256];}
  __syncthreads();
  if(tid<128) {sX[tid] += sX[tid + 128];}
  __syncthreads();
  if(tid<64) { sX[tid] += sX[tid + 64];}
  __syncthreads();
  if(tid<32){
    warpReduce(sX, tid);
  }

  __syncthreads();

  if(tid==0){
    dst[block] = sX[tid];
  }

}
#endif

#if defined (CUARCH) && (CUARCH<20)
__global__ void dot4_2048(knntype *dst, knntype *src, int attributes){

  __shared__ knntype sX[1024];

  int tid = threadIdx.x;
  int block = blockIdx.x;
  knntype res = 0;
  int offset = block * 2048 + tid;

  src += offset;

  knntype tmp = src[0];
  sX[tid] = tmp * tmp;
  tmp = src[512];
  sX[tid+512] = tmp*tmp;
  tmp = src[1024];

  __syncthreads();

  __syncthreads();
  if(tid<512){sX[tid] += sX[tid + 512];}
  __syncthreads();
  if(tid<256) {sX[tid] += sX[tid + 256];}
  __syncthreads();
  if(tid<128) {sX[tid] += sX[tid + 128];}
  __syncthreads();
  if(tid<64) { sX[tid] += sX[tid + 64];}
  __syncthreads();
  if(tid<32){
    warpReduce(sX, tid);
  }

  __syncthreads();


  res = sX[tid];


  sX[tid] = tmp*tmp;
  tmp = src[1024+512];
  sX[tid+512] = tmp*tmp;

  __syncthreads();
  if(tid<512){sX[tid] += sX[tid + 512];}
  __syncthreads();
  if(tid<256) {sX[tid] += sX[tid + 256];}
  __syncthreads();
  if(tid<128) {sX[tid] += sX[tid + 128];}
  __syncthreads();
  if(tid<64) { sX[tid] += sX[tid + 64];}
  __syncthreads();
  if(tid<32){
    warpReduce(sX, tid);
  }

  __syncthreads();


  if(tid==0){
    dst[block] = res + sX[tid];
  }

}
#endif

void order_output(knntype *trg, knntype *src, int numQueries, int k, int numBlocks, int maxQueries){

  for(int i=0; i<k; i++){
    for(int b=0; b<numQueries; b += maxQueries){
      int queryBlock = min(maxQueries , numQueries - b);
      for(int j=0; j< queryBlock; j++){
        trg[i*numQueries + b + j] = src[b*k + i*queryBlock + j];
      }
    }
  }


}

__global__ void gpu_order_output(knntype *trg, knntype *src, int numQueries, int k, int numBlocks, int maxQueries){

  int bqid = blockIdx.x * maxQueries + threadIdx.x;

  int qid = blockIdx.y * maxQueries + bqid;

  if(qid<numQueries && bqid < maxQueries){
    for(int i=0; i<k; i++){
      trg[k*numQueries + qid] = src[maxQueries*k + bqid];
    }
  }

}


// Coalesced transpose with no bank conflicts

__global__ void transposeNoBankConflicts(knntype *odata, knntype *idata, int width, int height, int nreps)
{
  __shared__ knntype tile[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}

// coalesced transpose (with bank conflicts)

__global__ void transposeCoalesced(knntype *odata, knntype *idata, int width, int height, int nreps)
{
  __shared__ knntype tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
  }
}


void transpose_naive(knntype *trg, knntype *src, int N, int Q){


  for(int i=0; i<N; i++){
    for(int j=0; j<Q; j++){
      trg[j*N+i] = src[i*Q+j];
    }
  }

}


void* ptranspose(void* argt){

  knntype *trg = ((thread_arg*)argt)->trg;
  knntype *src = ((thread_arg*)argt)->src;
  int N = ((thread_arg*)argt)->N;
  int Q = ((thread_arg*)argt)->Q;


  //printf("N=%d, Q=%d\n", N, Q);

  for(int i=0; i<N; i++){
    for(int j=0; j<Q; j++){
      trg[j*N+i] = src[i*Q+j];
    }
  }

}

void pdist_Q(knntype *dist, knntype *data, knntype *query, knntype *dotp, int objects, int attributes, int numQueries, cublasHandle_t handle,CUstream str, dotKernel *dotPKernel){

  knntype alpha = -2;
  knntype beta = 0;

  //Compute Q*D'
  cublasSetStream(handle, str);

#ifdef __DOUBLE__
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, numQueries, objects, attributes, &alpha, query, attributes, data, attributes, &beta, dist, numQueries);
#else
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, numQueries, objects, attributes, &alpha, query, attributes, data, attributes, &beta, dist, numQueries);
#endif

  //dim3 crpthreads(DIMENSIONS , 1);
  int nblocks = (objects > MAXBLOCKS) ? MAXBLOCKS : objects;
  dim3 crpgrid(nblocks, 1);
  int P = (int)ceil((float)objects/(float)MAXBLOCKS);

  dotPKernel->dotCrp<<<crpgrid, dotPKernel->nthreadsCrp, dotPKernel->externShared*sizeof(knntype), str>>>(dotp, data, objects, attributes, P);


  if(numQueries<=BLOCKSIZE){

    int thready = (int)ceil((knntype)BLOCKSIZE/(knntype)numQueries);
    dim3 threads(numQueries, thready);
    int block = (int)ceil((knntype)objects/(knntype)thready);
    int elempt = 2*(int)ceil((knntype)block / (knntype)MAXBLOCKS);
    int rblock = (int)ceil((knntype)block / (knntype)elempt);
    dim3 grid(rblock, 1);

    pdot<<<grid, threads, 0, str>>>(dist, dotp, objects, numQueries, elempt);
  }
  else{
    int thready = 1;
    dim3 threads(BLOCKSIZE, thready);
    int block = objects;

    int elempt = 2*(int)ceil((knntype)block / (knntype)MAXBLOCKS);
    int rblock = (int)ceil((knntype)block / (knntype)elempt);
    dim3 grid(rblock, (int)ceil((knntype)numQueries / (knntype)BLOCKSIZE));

    pdot_v2<<<grid, threads, 0, str>>>(dist, dotp, objects, numQueries, elempt);
  }


}


void pdist_N(knntype *dist, knntype *data, knntype *query, knntype *dotp, int objects, int attributes, int numQueries, cublasHandle_t handle, CUstream str){

  knntype alpha = -2;
  knntype beta = 0;

  //Compute D*Q'
  cublasSetStream(handle, str);

#ifdef __DOUBLE__
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, objects, numQueries, attributes, &alpha, data, objects, query, attributes, &beta, dist, objects);
#else
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, objects, numQueries, attributes, &alpha, data, objects, query, attributes, &beta, dist, objects);
#endif


  dim3 threads(BLOCKSIZE, 1);
  int block = (int)ceil((knntype)objects/(knntype)BLOCKSIZE);
  dim3 grid(block, 1);

  pdot<<<grid, threads, 0, str>>>(dist, dotp, objects, numQueries);

}



void pdist_NT(knntype *dist, knntype *data, knntype *query, knntype *dotp, int objects, int attributes,int numQueries, cublasHandle_t handle, CUstream str, dotKernel *dotPKernel){
  printf("%s\n", __func__);
  printf("objects=%d numQueries=%d attributes=%d\n", objects, numQueries, attributes);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed;
  cudaError_t err;

  knntype alpha = -2;
  knntype beta = 0;


  cublasSetStream(handle, str);

  cudaEventRecord(start);
#ifdef __DOUBLE__
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, objects, numQueries, attributes, &alpha, data, attributes, query, attributes, &beta, dist, objects);
#else
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, objects, numQueries, attributes, &alpha, data, attributes, query, attributes, &beta, dist, objects);
#endif
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  printf("cublas*gemm %.3f ms\n", elapsed);

  //dim3 crpthreads(DIMENSIONS , 1);
  int nblocks = (objects > MAXBLOCKS) ? MAXBLOCKS : objects;
  dim3 crpgrid(nblocks, 1);
  int P = (int)ceil((float)objects/(float)MAXBLOCKS);

  cudaEventRecord(start);
  dotPKernel->dotCrp<<<crpgrid, dotPKernel->nthreadsCrp, dotPKernel->externShared*sizeof(knntype), str>>>(dotp, data, objects, attributes, P);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  printf("dotPKernel->dotCrp %.3f ms\n", elapsed);

  dim3 threads(BLOCKSIZE, 1);
  int block = (int)ceil((knntype)objects/(knntype)BLOCKSIZE);
  dim3 grid(block, 1);

  cudaEventRecord(start);
  pdot<<<grid, threads, 0, str>>>(dist, dotp, objects, numQueries);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  printf("pdot %.3f ms\n", elapsed);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


__global__ void kernel_l1(knntype *dist, knntype *data, knntype *query, int objects, int attributes, int numQueries, int P){


  //int  point = blockIdx.x * P;

}


void pdist_l1(knntype *dist, knntype *data, knntype *query, knntype *dotp, int objects, int attributes, int numQueries, cublasHandle_t handle, CUstream str){



}



#endif
