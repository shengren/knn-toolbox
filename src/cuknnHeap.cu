#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "cuda.h"
#include "cublas.h"
#include "cublas_v2.h"
#include "../include/utils.h"

#ifndef TUNE
#define GRIDN 8
#define BLOCKSIZE 256
#else
#include "../include/utils.h"
#endif

#include <float.h>

#ifdef MATLABONLY
#include "mex.h"
#endif

#define MAXBLOCKS 32768
#define DIMENTIONS 128


void mergeResHeap(knntype *data, knntype *idx, int k, int Q, int numStreams){

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


__global__ void heap_initialize(knntype *h, int la, int ld, knntype val){

  int tid = BLOCKSIZE * blockIdx.x + threadIdx.x;

  h += blockIdx.y * la;

  if(tid < la){
    h[tid] = val;
  }

}


__global__ void heap_initialize(knntype *h, int la, int ld, knntype* val){

  int tid = BLOCKSIZE * blockIdx.x + threadIdx.x;

  h += blockIdx.y * la;
  val += tid;


  if(tid<la){
    h[tid] = val[0];
  }


}

__device__ void exchange(knntype *a, knntype *b){

  knntype tmp = a[0];
  a[0] = b[0];
  b[0] = tmp;
}

__global__ void final_unordered(knntype *data, knntype *index, knntype *srcIndex, int N, int k, int q){

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  knntype max, tmp;
  int max_ind = 0;

  data += tid;
  index += tid;

  max = data[0];

  if(tid < q){

    //Step 1
    for(int i=0; i<k; i++){
      tmp = data[i*q];
      if(max<tmp){
        max = tmp;
        max_ind = i;
      }
    }

    //Step 2
    for(int i=k; i<N; i++){
      tmp = data[i*q];

      if(max>tmp){
        max = tmp;
        data[max_ind*q] = tmp;
        index[max_ind*q] = srcIndex[i*q];

        for(int j=0; j<k; j++){
          tmp = data[j*q];
          if(max<tmp){
            max = tmp;
            max_ind = j;
          }
        }

      }

    }
  }

}

__device__ void heap_insertion(knntype *h, knntype *idx, knntype a, knntype newid, int qk, int Q){

  int c = 0;
  h[c] = a;
  idx[c] = newid;

  for(int j=0; j<qk-1; j++){
    int l = c << 1; int r = (l + 1);
    int li = l*Q; int ri = r*Q;
    int ci = c*Q;

    if(h[li] <= h[ri]){
      if(h[ci] < h[ri]){
        exchange(&h[ci], &h[ri]);
        exchange(&idx[ci], &idx[ri]);
        c = r;
      }
    }
    else{
      if(h[ci] < h[li]){
        exchange(&h[ci], &h[li]);
        exchange(&idx[ci], &idx[li]);
        c = l;
      }
    }

  }

}

__global__ void cpres(knntype *trg, knntype *src, int numQueries){

  int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if(tid<numQueries){
    trg[blockIdx.y*numQueries + tid] = src[blockIdx.y*numQueries + tid];
  }

}


__global__ void findMin(knntype* trg, knntype *dist, int P, int k, int numQueries){


  int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  dist += blockIdx.y * P * numQueries + tid;
  trg += blockIdx.y * numQueries + tid;

  if(tid<numQueries){

    knntype min = dist[0];
    for(int i=1; i<P; i++){
      min = fmin(min, dist[i*numQueries]);
    }

    trg[0] = min;

  }

}

__global__ void findMax(knntype *trg, int k, int numQueries){

  int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  trg += tid;

  if(tid<numQueries){

    knntype max = trg[0];

    for(int i=1; i<k; i++){
      max = fmax(max, trg[i*numQueries]);
    }
    trg[0] = max;
  }

}

__global__ void selection(knntype *dist, knntype *heap, knntype *index, int objects, int qk , int numQueries, int idOffset){



  int tid  = BLOCKSIZE * blockIdx.x + threadIdx.x;
  dist += tid + blockIdx.y * objects * numQueries;
  heap += tid + blockIdx.y * ((1<<qk)-1) * numQueries;
  index += tid + blockIdx.y * ((1<<qk)-1) * numQueries;

  int idxOffset = blockIdx.y * objects + idOffset;

  if(tid<numQueries){

    for(int i=0; i<objects; i++){

      knntype tmp = dist[i*numQueries];
      if(heap[0]>tmp){
        heap_insertion(heap, index, tmp, i + idxOffset, qk, numQueries);
      }

    }

  }

}


__global__ void selection_red(knntype *dist, knntype *heap, knntype* oldIds, knntype *index, int objects, int qk, int numQueries){

  int tid  = BLOCKSIZE * blockIdx.x + threadIdx.x;
  dist += tid + blockIdx.y * objects * numQueries;
  heap += tid + blockIdx.y * ((1<<qk)-1) * numQueries;
  index += tid + blockIdx.y * ((1<<qk)-1) * numQueries;
  oldIds += tid + blockIdx.y * objects * numQueries;

  if(tid<numQueries){

    for(int i=0; i<objects; i++){

      knntype tmp = dist[i*numQueries];
      if(heap[0]>tmp){
        heap_insertion(heap, index, tmp, oldIds[i*numQueries], qk, numQueries);
      }

    }
  }

}


void HeapSelect(knntype *dist, knntype *heap, knntype *index, int objects, int numQueries, int qk, CUstream str, int streamId){

  int block = (numQueries & (BLOCKSIZE-1)) ? numQueries / BLOCKSIZE + 1 : numQueries / BLOCKSIZE;
  dim3 Grid(block, 1);
  dim3 Threads(BLOCKSIZE, 1);

  //int idOffset = streamId*objects;
  int idOffset = streamId;

  //printf("offset :%d, objects: %d\n", idOffset, objects);

  selection<<< Grid, Threads, 0, str>>>(dist, heap, index, objects, qk, numQueries, idOffset);

  final_unordered<<<Grid, Threads, 0, str>>>(heap, index, index, (1<<qk)-1, (1<<(qk-1)), numQueries);

}

void HeapSelectGrid(knntype *dist, knntype *heap, knntype *index, int objects, int numQueries, int qk, CUstream str){

  knntype *Idxbuff;
  int block = (numQueries & (BLOCKSIZE-1)) ? numQueries / BLOCKSIZE + 1 : numQueries / BLOCKSIZE;
  dim3 Grid(block, GRIDN);
  dim3 Threads(BLOCKSIZE, 1);

  int fsize = ((1<<qk)-1)*GRIDN;
  int csize = (1<<qk) - 1;

  cudaMalloc((void**)&Idxbuff, fsize*numQueries*sizeof(knntype));

  //selection<<< Grid, Threads, 0, str>>>(dist, heap, Idxbuff, (objects / GRIDN), qk, numQueries);

  dim3 GridR(block, 1);
  dim3 ThreadsR(BLOCKSIZE, 1);

  int blockx = (numQueries & (BLOCKSIZE-1)) ? numQueries / BLOCKSIZE + 1  : numQueries / BLOCKSIZE;
  dim3 initGrid(blockx, fsize);
  dim3 cpGrid(blockx, csize);
  dim3 initThreads(BLOCKSIZE, 1);

  heap_initialize<<<initGrid, initThreads, 0, str>>>(dist, numQueries, fsize, FLT_MAX);

  selection_red<<<GridR, ThreadsR, 0, str>>>(heap, dist, Idxbuff, index, fsize, qk, numQueries);

  cpres<<<cpGrid, initThreads, 0, str>>>(heap, dist, numQueries);

  final_unordered<<<GridR, ThreadsR, 0, str>>>(heap, index, index, (1<<qk)-1, (1<<(qk-1)), numQueries);

  cudaFree(Idxbuff);
  

}

/* Not used */
void findUpperBound(knntype *val, knntype *dist, int objects, int numQueries, int k){

  int block = (int)ceil((float)numQueries / (float)BLOCKSIZE);
  dim3 grid1(block, k);
  dim3 threads1(BLOCKSIZE, 1);

  findMin<<<grid1, threads1>>>(val, dist, (objects/k), k, numQueries);

  findMax<<<block, BLOCKSIZE>>>(val, k, numQueries);

}

extern "C" void cuknnsHeap(knntype *dist, knntype *data, knntype *query, knntype *heap, knntype *index, knntype *dotp, int objects, int attributes, int numQueries, int qk, knntype *d_dotB, cublasHandle_t handle, CUstream str, knntimes *times, int streamId, distFunctParam *distFunc){

  
  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  

  int sizeofHeap = (1<<qk) - 1;

  int blockdp = (objects & (BLOCKSIZE-1)) ? objects / BLOCKSIZE + 1: objects / BLOCKSIZE;
  dim3 griddp(blockdp, 1);
  dim3 threadsdp(BLOCKSIZE, 1);

  cudaEventRecord(start, 0);

  distFunc->distF(dist, data, query, dotp, objects, attributes, numQueries, handle, str, &distFunc->dotP);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  times->dst_time += elapsedTime;

  int blockx = (numQueries & (BLOCKSIZE-1)) ? numQueries / BLOCKSIZE + 1: numQueries / BLOCKSIZE;
  dim3 initGrid(blockx, sizeofHeap);
  dim3 initThreads(BLOCKSIZE, 1);

  cudaEventRecord(start, 0);

  heap_initialize<<<initGrid, initThreads, 0, str>>>(heap, numQueries, sizeofHeap, FLT_MAX);

  HeapSelect(dist, heap, index, objects, numQueries, qk, str, streamId);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  times->srch_time += elapsedTime;

  int athreads = (numQueries>BLOCKSIZE) ? BLOCKSIZE : numQueries;
  dim3 threads2(athreads, 1);
  blockx = (numQueries & (BLOCKSIZE-1)) ? numQueries / BLOCKSIZE + 1: numQueries / BLOCKSIZE;
  int blocky = (1<<qk) - 1;
  dim3 grid2(blockx, blocky);

  pdot_v0<<<grid2, threads2>>>(heap, d_dotB, objects, numQueries);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}


extern "C" double gpuknnsHeap(knntype *query, knntype *data, knntype *values, knntype *indices, int objects, int numQueries, int attributes, int k, int numStreams){



  knntype *d_data, *d_query;
  knntype *d_dotp, *d_dist, *d_labels;
  knntype *d_heap, *d_dotB;
  size_t memory_free, memory_total;
  double TimeOut;

  knntimes TimesOut;
  TimesOut.dst_time = 0;
  TimesOut.srch_time = 0;
  TimesOut.dst_time = 0;

  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  /* initialize distance functions */
  distFunctParam dstfunc;
  dstfunc.distF = pdist_Q;
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


  cublasHandle_t handle;
  CUstream *stream = (CUstream*)malloc(numStreams*sizeof(CUstream));

  cublasCreate(&handle);

  for(int i=0; i<numStreams; i++){
    cuStreamCreate(&stream[i], 0);
  }

  int qk = (int)(log((float)k) / log(2.0)) + 1;
  int sizeofHeap = (1<<qk) - 1;

  knntype *outbuffDist = (knntype*)malloc(numStreams*numQueries*sizeofHeap*sizeof(knntype));
  knntype *outbuffIdx = (knntype*)malloc(numStreams*numQueries*sizeofHeap*sizeof(knntype));


  cudaMalloc((void**)&d_query, numQueries*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotB, numQueries*sizeof(knntype));

  cuMemGetInfo(&memory_free, &memory_total);
  int memory_req = (numQueries + attributes + sizeofHeap) * sizeof(float);

  int maxObjects = (int)ceil((float)memory_free*0.9 / (float)memory_req);

  maxObjects = (1 << (int)floor(log((double)maxObjects)/log(2)));

  int reqStreams = 1;

  int CorpusBlocks = (int)ceil((float)objects/(float)maxObjects);

  numStreams = max(numStreams, reqStreams);

  maxObjects = min(maxObjects, objects);


  knntype *outbuffDist2 = (knntype*)malloc(numStreams*CorpusBlocks*numQueries*sizeofHeap*sizeof(knntype));
  knntype *outbuffIdx2 = (knntype*)malloc(numStreams*CorpusBlocks*numQueries*sizeofHeap*sizeof(knntype));

  cudaMalloc((void**)&d_data, maxObjects*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotp, maxObjects*sizeof(knntype));
  cudaMalloc((void**)&d_dist, maxObjects*numQueries*sizeof(knntype));
  cudaMalloc((void**)&d_labels, numStreams*sizeofHeap*numQueries*sizeof(knntype));
  cudaMalloc((void**)&d_heap, numStreams*sizeofHeap*numQueries*sizeof(knntype));

  cudaEventRecord(start, 0);

  int fail = 0;

    cudaMemcpyAsync(d_query, query, numQueries*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[0]);

    dstfunc.dotP.dotQ<<<numQueries, dstfunc.dotP.nthreadsQ, dstfunc.dotP.externShared, stream[0]>>>(d_dotB, d_query, attributes);

  int count = 0;
  for(int ii=0, c=0; ii<objects; ii+=maxObjects, c++){

    int CorpusBlockSize = min(maxObjects, objects-ii);
    int StreamSize = CorpusBlockSize / numStreams;
    

    for(int jj=0; jj<numStreams; jj++){
      cudaMemcpyAsync(d_data + jj*StreamSize*attributes, data + ii*attributes + jj*StreamSize*attributes, StreamSize*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[jj]);
    }


    for(int jj=0; jj<numStreams; jj++){
      cuknnsHeap(d_dist+jj*StreamSize*numQueries, d_data+jj*StreamSize*attributes, d_query, d_heap+jj*sizeofHeap*numQueries, d_labels+jj*sizeofHeap*numQueries, d_dotp+jj*StreamSize, StreamSize, attributes, numQueries, qk, d_dotB, handle, stream[jj], &TimesOut, c*numStreams + jj, &dstfunc);    
    }

    for(int jj=0; jj<numStreams; jj++){
      cudaMemcpyAsync(outbuffDist + jj*k*numQueries + c*numStreams*k*numQueries, d_heap + jj*sizeofHeap*numQueries, k*numQueries*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
      cudaMemcpyAsync(outbuffIdx + jj*k*numQueries + c*numStreams*k*numQueries, d_labels + jj*sizeofHeap*numQueries, k*numQueries*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
    }
    count++;
  }


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);
  TimeOut = (fail==0) ? elapsedTime / 1000 : FLT_MAX;

  //printf("Time Elapsed: %f\n", TimeOut);

  for(int str=0; str<CorpusBlocks*numStreams; str++){
    transpose_naive(outbuffDist2+str*k*numQueries, outbuffDist+str*k*numQueries, k, numQueries);
    transpose_naive(outbuffIdx2+str*k*numQueries, outbuffIdx+str*k*numQueries, k, numQueries);
  }
  
  
  if(CorpusBlocks*numStreams>1){
    mergeResHeap(outbuffDist2, outbuffIdx2, k, numQueries, numStreams*CorpusBlocks);
  }
  
  memcpy(values, outbuffDist2, numQueries*k*sizeof(knntype));
  memcpy(indices, outbuffIdx2, numQueries*k*sizeof(knntype));


  for(int i=0; i<numStreams; i++){
    cuStreamDestroy(stream[i]);
  }
  

  free(outbuffDist);
  free(outbuffIdx);
  free(outbuffDist2);
  free(outbuffIdx2);
  cublasDestroy(handle);
  cudaFree(d_data);
  cudaFree(d_query);
  cudaFree(d_dotp);
  cudaFree(d_dist);
  cudaFree(d_labels);
  cudaFree(d_heap);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(stream);

  return(TimeOut);
}


