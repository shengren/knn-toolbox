
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "cuda.h"

#include "cublas.h"
#include "cublas_v2.h"
#include "../include/shared_functions.h"

#if defined (__DOUBLE__) && (CUARCH<20)
#define BLOCKSIZE 256
#define SHARED_SIZE_LIMIT 512
#else
#define BLOCKSIZE 512
#define SHARED_SIZE_LIMIT 1024
#endif

#define BLOCKSIZE_Q 128
#define DIMENTIONS 128

#define SINGLE_STREAM_BLOCKS 4

#define FORCE_SINGLE_STREAM -5

#ifndef MEMTEST

__device__ void Comparator(knntype& keyA, knntype& valA, knntype& keyB, knntype& valB, int dir){
  knntype t;
  if( (keyA > keyB) == dir ){
    t = keyA; keyA = keyB; keyB = t;
    t = valA; valA = valB; valB = t;
  }
}

__device__ void Comparator_elim(knntype& keyA, knntype& valA, knntype& keyB, knntype& valB, int dir){

  if( (keyA > keyB) == dir ){
    keyA = keyB;
    valA = valB;
  }

}

__global__ void bitonic_shared(knntype *DstKey, knntype *DstVal, knntype *SrcKey, knntype *SrcVal, int arrayLength, int objects, int queries, uint dir, int k, int qk, int idOffset){

  __shared__ knntype s_key[SHARED_SIZE_LIMIT];
  __shared__ knntype s_val[SHARED_SIZE_LIMIT];

  int tid = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;

  SrcKey += blockIdx.y*arrayLength;
  SrcVal += blockIdx.y*arrayLength;
  DstKey += blockIdx.y*arrayLength;
  DstVal += blockIdx.y*arrayLength;

  knntype* SrcKey_ptr = SrcKey;
  knntype* SrcVal_ptr = SrcVal;
  knntype* DstKey_ptr = DstKey;
  knntype* DstVal_ptr = DstVal;

  SrcKey_ptr += blockIdx.x*SHARED_SIZE_LIMIT + threadIdx.x;
  SrcVal_ptr += blockIdx.x*SHARED_SIZE_LIMIT + threadIdx.x;
  DstKey_ptr += blockIdx.x*SHARED_SIZE_LIMIT + threadIdx.x;
  DstVal_ptr += blockIdx.x*SHARED_SIZE_LIMIT + threadIdx.x;

  s_key[threadIdx.x + 0] = (tid<objects) ? SrcKey_ptr[0] : FLT_MAX;
  s_val[threadIdx.x + 0] = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + idOffset;
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT >> 1)] = ((tid + (SHARED_SIZE_LIMIT>>1))<objects) ? SrcKey_ptr[(SHARED_SIZE_LIMIT >> 1)] : FLT_MAX;
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT >> 1)] = blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x + (SHARED_SIZE_LIMIT >> 1) + idOffset;

  __syncthreads();

  uint ddd;
  uint pos;
  //SHARED_SIZE_LIMIT
  for(uint size = 2; size <= k; size <<= 1){
    //Bitonic merge
    ddd = dir^(threadIdx.x & (size >> 1)) != 0;
    for(uint stride = size >> 1; stride > 0; stride >>= 1){
      __syncthreads();
      pos = (threadIdx.x << 1) - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
    }
  }


  for(int obj = SHARED_SIZE_LIMIT >> 1; obj >= k; obj >>= 1){

    __syncthreads();

    // End of first part
    int bi = threadIdx.x >> qk;
    int li = threadIdx.x & (k-1);

    int pb = (obj >> qk) + ((bi + 1) & ((obj >> qk)-1));
    int prt = (pb << qk) + li;

    if(threadIdx.x<obj){
      Comparator_elim(s_key[threadIdx.x], s_val[threadIdx.x], s_key[prt], s_val[prt], 1);


      uint size = k;
      ddd = dir ^ ( (threadIdx.x & (size >> 1)) != 0 );
      for(int stride = size / 2; stride > 0; stride >>= 1){
	__syncthreads();
	pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
	Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
      }

    }
  }

  __syncthreads();

  if(threadIdx.x < k){
    DstKey[k*blockIdx.x + threadIdx.x] =  (blockIdx.x & 1) == 0 ? s_key[threadIdx.x] : s_key[k-threadIdx.x-1];
    DstVal[k*blockIdx.x + threadIdx.x] = (blockIdx.x & 1) == 0 ? s_val[threadIdx.x] : s_val[k-threadIdx.x-1];
  }

}

__global__ void bitonic_shared2(knntype *DstKey, knntype *DstVal, knntype *SrcKey, knntype *SrcVal, int arrayLength, int objects, int queries, uint dir, int k, int qk){

  __shared__ knntype s_key[SHARED_SIZE_LIMIT];
  __shared__ knntype s_val[SHARED_SIZE_LIMIT];


  int SIZE_LIMIT = blockDim.x << 1;
  int tid = blockIdx.x * SIZE_LIMIT + threadIdx.x;

  SrcKey += blockIdx.y*arrayLength;
  SrcVal += blockIdx.y*arrayLength;
  DstKey += blockIdx.y*arrayLength;
  DstVal += blockIdx.y*arrayLength;

  knntype* SrcKey_ptr = SrcKey;
  knntype* SrcVal_ptr = SrcVal;
  knntype* DstKey_ptr = DstKey;
  knntype* DstVal_ptr = DstVal;

  SrcKey_ptr += blockIdx.x*SIZE_LIMIT + threadIdx.x;
  SrcVal_ptr += blockIdx.x*SIZE_LIMIT + threadIdx.x;
  DstKey_ptr += blockIdx.x*SIZE_LIMIT + threadIdx.x;
  DstVal_ptr += blockIdx.x*SIZE_LIMIT + threadIdx.x;

  s_key[threadIdx.x + 0] = (tid<objects) ? SrcKey_ptr[0] : FLT_MAX;
  s_val[threadIdx.x + 0] = (tid<objects) ? SrcVal_ptr[0] : FLT_MAX;
  s_key[threadIdx.x + (SIZE_LIMIT >> 1)] = ((tid+(SIZE_LIMIT>>1))<objects) ? SrcKey_ptr[(SIZE_LIMIT >> 1)] : FLT_MAX;
  s_val[threadIdx.x + (SIZE_LIMIT >> 1)] = ((tid+(SIZE_LIMIT>>1))<objects) ? SrcVal_ptr[(SIZE_LIMIT >> 1)] : FLT_MAX;

  __syncthreads();

  uint ddd;
  uint pos;

  for(int obj = SIZE_LIMIT >> 1; obj >= k; obj >>= 1){

    __syncthreads();

    // End of first part
    int bi = threadIdx.x >> qk;
    int li = threadIdx.x & (k-1);

    int pb = (obj >> qk) + ((bi + 1) & ((obj >> qk)-1));
    int prt = (pb << qk) + li;

    if(threadIdx.x<obj){
      Comparator_elim(s_key[threadIdx.x], s_val[threadIdx.x], s_key[prt], s_val[prt], 1);

      uint size = k;
      ddd = dir ^ ( (threadIdx.x & (size >> 1)) != 0 );
      for(int stride = size / 2; stride > 0; stride >>= 1){
        __syncthreads();
        pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
      }

    }
  }

  __syncthreads();

  if(threadIdx.x < k){
    DstKey[k*blockIdx.x + threadIdx.x] =  (blockIdx.x & 1) == 0 ? s_key[threadIdx.x] : s_key[k-threadIdx.x-1];
    DstVal[k*blockIdx.x + threadIdx.x] = (blockIdx.x & 1) == 0 ? s_val[threadIdx.x] : s_val[k-threadIdx.x-1];
  }

}


__global__ void relloc(knntype *DstKey, knntype *DstVal, knntype *SrcKey, knntype *SrcVal, int objects, int k){

  DstKey += blockIdx.x * k + threadIdx.x;
  DstVal += blockIdx.x * k + threadIdx.x;
  SrcKey += blockIdx.x * objects + threadIdx.x;
  SrcVal += blockIdx.x * objects + threadIdx.x;

  if(threadIdx.x < k){
    DstKey[0] = SrcKey[0];
    DstVal[0] = SrcVal[0];
  }

}


void BitonicSelect(knntype *DstKey, knntype *DstVal, knntype *SrcKey, knntype *SrcVal, knntype *buffkey, knntype *buffval, int objects, int queries, int k, int qk, CUstream str, int streamId){

  int numObjects = objects;

  knntype *tmpkey1 = SrcKey;
  knntype *tmpkey2 = buffkey;
  knntype *tmpval1 = SrcVal;
  knntype *tmpval2 = buffval;

  knntype *tmpV;
  knntype *tmpK;

  dim3 threads(SHARED_SIZE_LIMIT / 2, 1);
  int grd = (objects & (SHARED_SIZE_LIMIT-1)) ? objects / SHARED_SIZE_LIMIT + 1 : objects / SHARED_SIZE_LIMIT;
  dim3 grid(grd, queries);

  int idOffset = streamId*objects;

  bitonic_shared<<<grid, threads, 0, str>>>(buffkey, buffval, SrcKey, SrcVal, numObjects, objects, queries, 1, k, qk, idOffset);

  objects = grd*k;
  int robjects = objects;
  objects = (objects & (SHARED_SIZE_LIMIT-1)) ? (objects / SHARED_SIZE_LIMIT + 1)*SHARED_SIZE_LIMIT : (objects / SHARED_SIZE_LIMIT)*SHARED_SIZE_LIMIT;

  while(robjects > k){

    int blockSize = SHARED_SIZE_LIMIT<objects ? SHARED_SIZE_LIMIT : objects;
    dim3 threadsp(blockSize/2, 1);
    dim3 gridp(objects / blockSize, queries);

    bitonic_shared2<<<gridp, threadsp, 0, str>>>(tmpkey1, tmpval1, tmpkey2, tmpval2, numObjects, robjects, queries, 1, k, qk);

    tmpK = tmpkey1; tmpkey1 = tmpkey2; tmpkey2 = tmpK;
    tmpV = tmpval1; tmpval1 = tmpval2; tmpval2 = tmpV;

    objects = k*(objects / blockSize);
    robjects = objects;
    objects = (objects & (SHARED_SIZE_LIMIT-1)) ? (objects / SHARED_SIZE_LIMIT + 1)*SHARED_SIZE_LIMIT : (objects / SHARED_SIZE_LIMIT)*SHARED_SIZE_LIMIT;

  }


  dim3 threads_relloc(k, 1);
  dim3 grid_relloc(queries,1);

  relloc<<<grid_relloc, threads_relloc, 0, str>>>(DstKey, DstVal, tmpkey2, tmpval2, numObjects, k);

}


__global__ void initialize_index_B(knntype* data, int objects, int numQueries){

  int tid = threadIdx.x + blockIdx.x * blockDim.x;


  if(tid<objects){
#pragma unroll 2
    for(int i=0; i<numQueries; i++){
      data[i*objects + tid] = tid;
    }
  }
}
/* Test function test function currently no used */
extern "C" void cuknnsBitonic(knntype *dist, knntype *data, knntype *query, knntype *index, knntype *dotp, knntype *d_dotB, knntype *distbuff, knntype *idxbuff, int objects, int attributes, int numQueries, int k, cublasHandle_t handle, CUstream str, knntimes* times, int strId){

  float tmp_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int qk = (int)(log((float)k) / log(2.0));

  dim3 inThreads(BLOCKSIZE, 1);
  int block = (objects & (BLOCKSIZE-1)) ? objects / BLOCKSIZE + 1 : objects / BLOCKSIZE;
  dim3 inGrid(block, 1);

  cudaEventRecord(start, 0);

  initialize_index_B<<<inGrid, inThreads, 0, str>>>(index, objects, numQueries);

  pdist_N(dist, data, query, dotp, objects, attributes, numQueries, handle, str);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&tmp_time, start, stop);

  times->dst_time += tmp_time;

  cudaEventRecord(start, 0);

  BitonicSelect(dist, index, dist, index, distbuff, idxbuff, objects, numQueries, k, qk, str, strId);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&tmp_time, start, stop);

  times->srch_time += tmp_time;

  switch(attributes){

  case 50:
    //dot4_50<<<numQueries, 50, 0, str>>>(d_ditB, quary);
  case 128:
    //dot4<<<numQueries, DIMENTIONS, 0, str>>>(d_dotB, query);
    break;
  case 1024:
    //dot4_1024<<<numQueries, 512, 0, str>>>(d_dotB, query);
    break;
  case 2048:
#if defined (CUARCH) && (CUARCH>=20)
    //dot4_2048<<<numQueries, 1024, 0, str>>>(d_dotB, query);
#endif
#if defined(CUARCH) && (CUARCH<20)
    //dot4_2048<<<numQueries, 512, 0, str>>>(d_dotB, query);
#endif
    break;
  }

  dim3 threads2(k, 1);
  dim3 grid2(numQueries, 1);

  dot3<<<grid2, threads2, 0, str>>>(dist, d_dotB);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/* KNNS using TBiS */
extern "C" void cuknnsBitonicSTR(knntype *dist, knntype *data, knntype *query, knntype *index, knntype *dotp, knntype *d_dotB, knntype *distbuff, knntype *idxbuff, int objects, int attributes, int numQueries, int k, cublasHandle_t handle, CUstream str, knntimes* times, int strId, distFunctParam *distFunc){
  printf("%s\n", __func__);

  float tmp_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  int qk = (int)(log((float)k) / log(2.0));


  dim3 inThreads(BLOCKSIZE, 1);
  int block = (objects & (BLOCKSIZE-1)) ? objects / BLOCKSIZE + 1 : objects / BLOCKSIZE;
  dim3 inGrid(block, 1);

  //cudaEventRecord(start, str);

  distFunc->distF(dist, data, query, dotp, objects, attributes, numQueries, handle, str, &distFunc->dotP);

  //cudaEventRecord(stop, str);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&tmp_time, start, stop);
  //times->dst_time = tmp_time;
  //printf("distFunc->distF %.3f ms\n", tmp_time);

  cudaEventRecord(start, str);

  BitonicSelect(dist, index, dist, index, distbuff, idxbuff, objects, numQueries, k, qk, str, strId);

  cudaEventRecord(stop, str);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tmp_time, start, stop);
  times->srch_time += tmp_time;
  times->srch_time = tmp_time;
  printf("BitonicSelect %.3f ms\n", tmp_time);

  dim3 threads2(k, 1);
  dim3 grid2(numQueries, 1);

  cudaEventRecord(start);
  dot3<<<grid2, threads2, 0, str>>>(dist, d_dotB);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&tmp_time, start, stop);
  printf("dot3 %.3f ms\n", tmp_time);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}


void mergeResBitonic(knntype *data, knntype *idx, int k, int Q, int numStreams){
  printf("%s\n", __func__);

  for(int s=0; s<Q; s++){

    knntype cmax = -FLT_MAX;
    int maxid = 0;
    for(int i=0; i<k; i++){
      knntype tmp = data[s*k+i];
      if(tmp>cmax){
        cmax = tmp;
        maxid = i;
      }
    }

    for(int i=1; i<numStreams; i++){
      for(int j=0; j<k; j++){
        knntype tmp = data[s*k + i*Q*k + j];
        if(tmp<cmax){
          data[s*k + maxid] = tmp;
          idx[s*k + maxid] = idx[s*k + i*Q*k + j];
          //max = data[s*k];
          cmax = -FLT_MAX;
          for(int p=0; p<k; p++){
            knntype tmp2 = data[s*k + p];
            if(tmp2>cmax){
              cmax = tmp2;
              maxid = p;
            }
          }

        }
      }
    }

  }

}

#endif

#ifndef MEMTEST
extern "C" double gpuknnsBitonic(knntype *query, knntype *data, knntype *values, knntype *indices, int objects, int numQueries, int attributes, int k, int numStreams){
#else
extern "C" double gpuknnsBitonicMemTest(knntype *query, knntype *data, knntype *values, knntype *indices, int objects, int numQueries, int attributes, int k, int numStreams){
#endif
  printf("%s\n", __func__);

  knntype *d_data, *d_query;
  knntype *d_dotp, *d_dist, *d_labels;

  size_t memory_free, memory_total;
  double TimeOut;
  knntimes TimesOut;

  cuMemGetInfo(&memory_free, &memory_total);

  TimesOut.dst_time = 0;
  TimesOut.srch_time = 0;
  TimesOut.knn_time = 0;

  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cublasHandle_t handle;

  cublasCreate(&handle);

  cudaMalloc((void**)&d_query, numQueries*attributes*sizeof(knntype));

  printf("attributes=%d\n", attributes);

  /* initialize distance functions */
  distFunctParam dstfunc;
  dstfunc.distF = pdist_NT;
  switch(attributes){
  case 50:
    dstfunc.dotP.dotQ = &dot4_50;
    dstfunc.dotP.nthreadsQ = 32;
    dstfunc.dotP.dotCrp = &crpdot2_50;
    dstfunc.dotP.nthreadsCrp = 32;
    dstfunc.dotP.externShared = 0;
    break;
  case 128:
    dstfunc.dotP.dotQ = &dot4;
    dstfunc.dotP.nthreadsQ = 128;
    dstfunc.dotP.dotCrp = &crpdot2_128;
    dstfunc.dotP.nthreadsCrp = 128;
    dstfunc.dotP.externShared = 0;
    break;
  case 1024:
    dstfunc.dotP.dotQ = &dot4_1024;
    dstfunc.dotP.nthreadsQ = 512;
    dstfunc.dotP.dotCrp = &crpdot2_1024;
    dstfunc.dotP.nthreadsCrp = 512;
    dstfunc.dotP.externShared = 0;
    break;
  case 2048:
    dstfunc.dotP.dotQ = &dot4_2048;
    dstfunc.dotP.nthreadsQ = 512;
    dstfunc.dotP.dotCrp = &crpdot2_2048;
    dstfunc.dotP.nthreadsCrp = 512;
    dstfunc.dotP.externShared = 0;
    break;
  default:
    dstfunc.dotP.dotQ = &dot4_gen;
    dstfunc.dotP.nthreadsQ = ceil(log2((float)attributes));
    dstfunc.dotP.dotCrp = &crpdot2_gen;
    dstfunc.dotP.nthreadsCrp = dstfunc.dotP.nthreadsQ;
    dstfunc.dotP.externShared = dstfunc.dotP.nthreadsQ;
  }

  /* calculate the number of streams */
  int memory_req = (4*numQueries + attributes)*sizeof(knntype);

  int maxObjects = (int)ceil((float)memory_free*0.9 / (float)memory_req);

  maxObjects = (1 << (int)floor(log((double)maxObjects)/log(2)));

#ifndef MEMTEST
  int reqStreams = (maxObjects < objects) ? 2 : 1;
#endif
#ifdef MEMTEST
  int reqStreams = 1;
#endif

  int CorpusBlocks = (int)ceil((float)objects/(float)maxObjects);


  int blocksPstream = (numStreams == FORCE_SINGLE_STREAM) ? SINGLE_STREAM_BLOCKS : 1;
  numStreams = (numStreams == FORCE_SINGLE_STREAM) ? 1 : max(numStreams, reqStreams);

  maxObjects = min(maxObjects, objects);

  /*Initialize Streams */
  CUstream *stream = (CUstream*)malloc(numStreams*sizeof(CUstream));

  for(int i=0; i<numStreams; i++){
    cuStreamCreate(&stream[i], 0);
  }

  printf("numStreams=%d maxObjects=%d\n", numStreams, maxObjects);

  /* Initialize memory */
  knntype *outbuffDist, *outbuffIdx;
  cudaMallocHost((void**)&outbuffDist, blocksPstream*numStreams*CorpusBlocks*numQueries*k*sizeof(knntype));
  cudaMallocHost((void**)&outbuffIdx, blocksPstream*numStreams*CorpusBlocks*numQueries*k*sizeof(knntype));

  cudaMalloc((void**)&d_data, maxObjects*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotp, maxObjects*sizeof(knntype));

  cudaMalloc((void**)&d_dist, maxObjects*numQueries*sizeof(knntype));
  cudaMalloc((void**)&d_labels, maxObjects*numQueries*sizeof(knntype));

  knntype *d_dotB, *distbuff, *idxbuff;
  cudaMalloc((void**)&d_dotB, numQueries*sizeof(knntype));
  cudaMalloc((void**)&distbuff, numQueries*maxObjects*sizeof(knntype));
  cudaMalloc((void**)&idxbuff, numQueries*maxObjects*sizeof(knntype));

  cudaMemcpyAsync(d_query, query, numQueries*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[0]);

  /*compute the dot product of the queries*/
  cudaEventRecord(start);
  dstfunc.dotP.dotQ<<<numQueries, dstfunc.dotP.nthreadsQ, dstfunc.dotP.externShared, stream[0]>>>(d_dotB, d_query, attributes);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("dstfunc.dotP.dotQ %.3f ms\n", elapsedTime);

  int fail = 0;
  //float TotalDstTime = 0;
  //float TotalSeachTime = 0;
  //float TotalCompTime = 0;
  //cudaEventRecord(start, 0);


  for(int ii=0, c = 0; ii<objects; ii+=maxObjects, c++){

    int CorpusBlockSize = min(maxObjects, objects-ii);
    int StreamSize = CorpusBlockSize / numStreams;

    for(int jj=0; jj<numStreams; jj++){
      cudaMemcpyAsync(d_data + jj*StreamSize*attributes, data + ii*attributes + jj*StreamSize*attributes, StreamSize*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[jj]);
    }

#ifndef MEMTEST

    for(int jj=0; jj<numStreams; jj++){
      cuknnsBitonicSTR(d_dist + jj*StreamSize*numQueries, d_data + jj*StreamSize*attributes, d_query, d_labels + jj*StreamSize*numQueries, d_dotp+jj*StreamSize, d_dotB, distbuff + jj*StreamSize*numQueries, idxbuff + jj*StreamSize*numQueries, StreamSize, attributes, numQueries, k, handle, stream[jj], &TimesOut, c*numStreams + jj, &dstfunc);

    }
#endif

    for(int jj=0; jj<numStreams; jj++){
      cudaMemcpyAsync(outbuffDist + jj*k*numQueries + c*numStreams*k*numQueries, d_dist + jj*StreamSize*numQueries, k*numQueries*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
      cudaMemcpyAsync(outbuffIdx + jj*k*numQueries + c*numStreams*k*numQueries, d_labels + jj*StreamSize*numQueries, k*numQueries*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
    }

  }

  cuCtxSynchronize();

  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);

  //cudaEventElapsedTime(&elapsedTime, start, stop);

  //TimesOut.knn_time = (fail==0) ? elapsedTime / 1000 : FLT_MAX;
  //TimeOut = TimesOut.knn_time;

  //printf("Time Elapsed: %f\n", TimeOut);

  int ss = numStreams;
  if(ss*CorpusBlocks>1){
    mergeResBitonic(outbuffDist, outbuffIdx, k, numQueries, ss*CorpusBlocks);
  }

  memcpy(values, outbuffDist, k*numQueries*sizeof(knntype));
  memcpy(indices, outbuffIdx, k*numQueries*sizeof(knntype));


  for(int i=0; i<numStreams; i++){
    cuStreamDestroy(stream[i]);
  }

  cudaFree(d_dotB);
  cudaFree(distbuff);
  cudaFree(idxbuff);

  cudaFreeHost(outbuffDist);
  cudaFreeHost(outbuffIdx);
  cublasDestroy(handle);
  cudaFree(d_data);
  cudaFree(d_query);
  cudaFree(d_dotp);
  cudaFree(d_dist);
  cudaFree(d_labels);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(stream);


  return(TimeOut);

}


/* Under development */
#ifndef MEMTEST
 extern "C" float gpuknnLSH(knntype *query, knntype *data, knntype *values, knntype *indices, knntype *dp, int objects, int numQueries, int attributes, int k, int numStreams, int *bucketSize, int *query_offsets, int numClusters, int *query_sizes){
#else
   extern "C" float gpuknnLSHmemtest(knntype *query, knntype *data, knntype *values, knntype *indices, knntype *dp, int objects, int numQueries, int attributes, int k, int numStreams, int *bucketSize,  int *query_offsets, int numClusters, int *query_sizes){
#endif
  knntype *d_data, *d_query;
  knntype *d_dotp, *d_dist, *d_labels;

  size_t memory_free, memory_total;
  double TimeOut;
  knntimes TimesOut;

  cuMemGetInfo(&memory_free, &memory_total);

  TimesOut.dst_time = 0;
  TimesOut.srch_time = 0;
  TimesOut.knn_time = 0;

  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cublasHandle_t handle;

  cublasCreate(&handle);

  cudaMalloc((void**)&d_query, numQueries*attributes*sizeof(knntype));

  /* calculate the number of streams */
  int memory_req = (4*numQueries + attributes)*sizeof(knntype);
  //int memory_req = (4*numStreams + attributes)*sizeof(knntype);


  int maxObjects = (int)ceil((float)memory_free*0.9 / (float)memory_req);

  maxObjects = (1 << (int)floor(log((double)maxObjects)/log(2)));

  /*
#ifndef MEMTEST
  int reqStreams = (maxObjects < objects) ? 2 : 1;
#endif
#ifdef MEMTEST
  int reqStreams = 1;
#endif
  */

  int CorpusBlocks = (int)ceil((float)objects/(float)maxObjects);

  //int StreamingEnable = 1;
  int blocksPstream = (numStreams == FORCE_SINGLE_STREAM) ? SINGLE_STREAM_BLOCKS : 1;
  numStreams = (numStreams == FORCE_SINGLE_STREAM) ? 1 : numStreams;

  //printf("General inter:\n");
  //printf("Free memory: %d\n", memory_free);
  //printf("Objects : %d\n", objects);
  //printf("max Corpus size Size: %d\n", maxObjects);
  //printf("Blocks of the Corpus: %d\n", CorpusBlocks);
  //printf("numStreams = %d\n", numStreams);
  //printf("Blocks Per Stream: %d\n", blocksPstream);

  //maxObjects = min(maxObjects, objects);
  //maxObjects = min(maxObjects, (int)ceil(objects/1024)*1024);

  printf("New maxObjets: %d\n", maxObjects);

  /*Initialize Streams */
  CUstream *stream = (CUstream*)malloc(numStreams*sizeof(CUstream));

  for(int i=0; i<numStreams; i++){
    cuStreamCreate(&stream[i], 0);
  }

  /* Initialize memory */
  knntype *outbuffDist, *outbuffIdx;
  cudaMallocHost((void**)&outbuffDist, blocksPstream*numStreams*CorpusBlocks*numQueries*k*sizeof(knntype));
  cudaMallocHost((void**)&outbuffIdx, blocksPstream*numStreams*CorpusBlocks*numQueries*k*sizeof(knntype));
  //cudaMallocHost((void**)&outbuffDist, numQueries*k*sizeof(knntype));
  //cudaMallocHost((void**)&outbuffIdx, numQueries*k*sizeof(knntype));

  cudaMalloc((void**)&d_data, maxObjects*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotp, maxObjects*sizeof(knntype));

  cudaMalloc((void**)&d_dist, maxObjects*numQueries*sizeof(knntype));
  cudaMalloc((void**)&d_labels, maxObjects*numQueries*sizeof(knntype));
  //cudaMalloc((void**)&d_dist, maxObjects*numStreams*sizeof(knntype));
  //cudaMalloc((void**)&d_labels, maxObjects*numStreams*sizeof(knntype));


  knntype *d_dotB, *distbuff, *idxbuff;
  cudaMalloc((void**)&d_dotB, numQueries*sizeof(knntype));
  cudaMalloc((void**)&distbuff, numQueries*maxObjects*sizeof(knntype));
  cudaMalloc((void**)&idxbuff, numQueries*maxObjects*sizeof(knntype));
  //cudaMalloc((void**)&distbuff, numStreams*maxObjects*sizeof(knntype));
  //cudaMalloc((void**)&idxbuff, numStreams*maxObjects*sizeof(knntype));


  cudaMemcpyAsync(d_query, query, numQueries*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[0]);

  /*compute the dot product of the queries*/
  switch(attributes){

  case 128:
    //dot4<<<numQueries, DIMENTIONS, 0, stream[0]>>>(d_dotB, d_query);
    break;
  case 1024:
    //dot4_1024<<<numQueries, 512, 0, stream[0]>>>(d_dotB, d_query);
    break;
  case 2048:
#if defined (CUARCH) && (CUARCH>=20)
    //dot4_2048<<<numQueries, 1024, 0, stream[0]>>>(d_dotB, d_query);
#endif
#if defined(CUARCH) && (CUARCH<20)
    //dot4_2048<<<numQueries, 512, 0, stream[0]>>>(d_dotB, d_query);
#endif
    break;
  }

  //printf("Starting Streaming\n");

  int fail = 0;
  //float TotalDstTime = 0;
  //float TotalSeachTime = 0;
  //float TotalCompTime = 0;
  cudaEventRecord(start, 0);

  int offset = 0;
  for(int ii=0, c = 0; ii<numClusters; ii+=numStreams, c++){

    //int CorpusBlockSize = min(maxObjects, objects-ii);
    int StreamSize = bucketSize[ii];
    //int offset = redSizes[ii];
    int pasedStreams = 0;
    int mOffset = (offset & ((objects>>1)-1));


    //for(int jj=0; jj<numStreams, pasedStreams<maxObjects; jj++){
    for(int jj=0; jj<numStreams; jj++){
      StreamSize = bucketSize[ii+jj];
      //printf("Query: %d, stream: %d, pasedStreams:%d, offset: %d, moffset: %d\n", ii+jj, jj, pasedStreams, offset+pasedStreams, mOffset);
      cudaMemcpyAsync(d_data + pasedStreams*attributes, data + mOffset*attributes + pasedStreams*attributes, StreamSize*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[jj]);
      cudaMemcpyAsync(d_dotp + pasedStreams, dp + mOffset + pasedStreams, StreamSize*sizeof(knntype), cudaMemcpyHostToDevice, stream[jj]);
      pasedStreams += StreamSize;
    }

#ifndef MEMTEST
    pasedStreams = 0;
    //for(int jj=0; jj<numStreams, pasedStreams<maxObjects; jj++){
    for(int jj=0; jj<numStreams; jj++){
      StreamSize = bucketSize[ii+jj];
      //cuknnsBitonicSTR(d_dist + pasedStreams*query_offsets[ii+jj], d_data + pasedStreams*attributes, d_query+query_offsets[ii+jj]*attributes, d_labels + pasedStreams*query_offsets[ii+jj], d_dotp+pasedStreams, d_dotB + query_offsets[ii+jj], distbuff + pasedStreams*query_offsets[ii+jj], idxbuff + pasedStreams*query_offsets[ii+jj], StreamSize, attributes, query_sizes[ii+jj], k, handle, stream[jj], &TimesOut, jj);
      pasedStreams += StreamSize;
    }
#endif
    pasedStreams = 0;
    //for(int jj=0; jj<numStreams, pasedStreams<maxObjects; jj++){
    for(int jj=0; jj<numStreams; jj++){
      StreamSize = bucketSize[ii+jj];
      cudaMemcpyAsync(outbuffDist + k*query_offsets[ii+jj], d_dist + pasedStreams*query_offsets[ii+jj], query_sizes[ii+jj]*k*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
      cudaMemcpyAsync(outbuffIdx + k*query_offsets[ii+jj], d_labels + pasedStreams*query_offsets[ii+jj], query_sizes[ii+jj]*k*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
      pasedStreams += StreamSize;
    }

    offset += pasedStreams;
  }


  cuCtxSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  printf("Exiting streaming... \n");

  cudaEventElapsedTime(&elapsedTime, start, stop);

  TimesOut.knn_time = (fail==0) ? elapsedTime / 1000 : FLT_MAX;
  TimeOut = TimesOut.knn_time;

#ifndef MEMTEST

  printf("Bitonic Search: N: %d, Q: %d, streams: %d, time : %f\n", objects, numQueries, numStreams, TimeOut);
  //printf("Computation Time: %f\n", (TimesOut.dst_time + TimesOut.srch_time) / 1000);
  printf("Time elapsed knns with Bitonic Search: %f\n", TimeOut);

#else
  printf("Data transfer: N: %d, Q: %d, time elapsed: %f\n", objects, numQueries, TimeOut);
#endif

  /*
#ifndef MEMTEST
  int ss = numStreams;
  printf("ss; %d\n", ss);
  if(ss*CorpusBlocks>1){
    mergeResBitonic(outbuffDist, outbuffIdx, k, numQueries, ss*CorpusBlocks);
  }
#endif
  */

  memcpy(values, outbuffDist, k*numQueries*sizeof(knntype));
  memcpy(indices, outbuffIdx, k*numQueries*sizeof(knntype));


  for(int i=0; i<numStreams; i++){
    cuStreamDestroy(stream[i]);
  }

  cudaFree(d_dotB);
  cudaFree(distbuff);
  cudaFree(idxbuff);

  cudaFreeHost(outbuffDist);
  cudaFreeHost(outbuffIdx);
  cublasDestroy(handle);
  cudaFree(d_data);
  cudaFree(d_query);
  cudaFree(d_dotp);
  cudaFree(d_dist);
  cudaFree(d_labels);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(stream);
  //cudaDeviceReset();

  return(TimeOut);
 }









