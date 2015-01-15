#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cublas.h"
#include "../include/cuknns.h"
#include "../include/shared_functions.h"
#include <sys/time.h>
#include <pthread.h>

#include "mex.h"
#define BLOCKSIZE 256
#define DIMENTIONS 128
//#define MAXQUERYBLOCK 256
//#define MAXQUERYBLOCK 32768
#define MAXQUERYBLOCK 16384
//#define MAXQUERYBLOCK 8192
//#define MAXQUERYBLOCK 4096


#define NTHREADS 8

typedef struct{
  knntype* data;
  knntype* idx;
  int k;
  int qth;
  int Q;
  int numStreams;
}thread_mg;

char* errMsg = "Only 128 dimentional data curentrly suported\n";

__global__ void heap_init(knntype *h, int la, int ld){

  int tid = BLOCKSIZE * blockIdx.x + threadIdx.x;

  h += blockIdx.y * la;

  if(tid < la){
    h[tid] = FLT_MAX;
  }

}

void mex_order_output(knntype *trg, knntype *src, int numQueries, int k, int numBlocks, int maxQueries){

  for(int i=0; i<k; i++){
    for(int b=0; b<numQueries; b += maxQueries){
      int queryBlock = min(maxQueries , numQueries - b);
      for(int j=0; j< queryBlock; j++){
        trg[i*numQueries + b + j] = src[b*k + i*queryBlock + j];
      }
    }
  }

}


void* pmergeRes(void* argm){

  knntype *data = ((thread_mg*)argm)->data;
  knntype *idx = ((thread_mg*)argm)->idx;
  int k = ((thread_mg*)argm)->k;
  int qth = ((thread_mg*)argm)->qth;
  int Q = ((thread_mg*)argm)->Q;
  int numStreams = ((thread_mg*)argm)->numStreams;

  for(int s=0; s<qth; s++){

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



void mergeRes(knntype *data, knntype *idx, int k, int Q, int numStreams){

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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

  /*=========MATLAB pointers=========*/
  knntype *data, *dp;
  knntype *query;
  double *TimeOut;
  size_t memory_free, memory_total;
  knntype *outbuffDist, *outbuffDist2;
  knntype *outbuffIdx, *outbuffIdx2;
  knntimes TimesOut;

  TimesOut.dst_time = 0;
  TimesOut.srch_time = 0;
  TimesOut.knn_time = 0;

  struct timeval startwtime, endwtime;
  double serial_time = 0;
  double transpose_time = 0;


  //pthread_t *cputhreads;
  pthread_t cputhreads[NTHREADS];
  //thread_arg *argt;
  thread_mg argm[NTHREADS];
  pthread_attr_t attr;
  void *status;
  int rc, t;

  /*========= Devece pointers ======*/
  knntype *d_data, *d_query, *d_labels, *d_dist, *d_dotp;
  knntype *d_heap;

  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cublasHandle_t handle;

  cublasCreate(&handle);

  query = (knntype*)mxGetData(prhs[0]);
  int numQueries = mxGetN(prhs[0]);
  data = (knntype*)mxGetData(prhs[1]);
  int objects = mxGetN(prhs[1]);
  int attributes = mxGetM(prhs[0]);
  int k = (int)(mxGetScalar(prhs[2]));
  int numStreams = (int)mxGetScalar(prhs[3]);


  int qk = (int)(log((float)k) / log(2.0)) + 1;
  int sizeofHeap = (1<<qk) - 1;

  int maxQueries = min(numQueries, MAXQUERYBLOCK);

  knntype *d_dotB;
  cudaMalloc((void**)&d_query, maxQueries*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotB, maxQueries*sizeof(knntype));

  cuMemGetInfo(&memory_free, &memory_total);
  int memory_req = (maxQueries + attributes + sizeofHeap) * sizeof(knntype);

  int maxObjects = (int)ceil((float)memory_free*0.9 / (float)memory_req);

  int reqStreams = 1;

  int CorpusBlocks = (int)ceil((float)objects/(float)maxObjects);

  numStreams = max(numStreams, reqStreams);
  maxObjects = min(maxObjects, objects);


  /* initialize distance functions */
  distFunctParam dstfunc;
  dstfunc.distF = pdist_Q;
  switch(attributes){
  case 50:
    dstfunc.dotP.dotQ = &dot4_50;
    dstfunc.dotP.nthreadsQ = 64;
    dstfunc.dotP.dotCrp = &crpdot2_50;
    dstfunc.dotP.nthreadsCrp = 64;
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
    dstfunc.dotP.nthreadsQ = (attributes>32) ? (1 << (int)ceil(log2((float)attributes))) : 32;
    dstfunc.dotP.dotCrp = &crpdot2_gen;
    dstfunc.dotP.nthreadsCrp = dstfunc.dotP.nthreadsQ;
    dstfunc.dotP.externShared = dstfunc.dotP.nthreadsQ;
  }

  /* initialize output */
#ifndef __DOUBLE__
  plhs[0] = mxCreateNumericMatrix(k, numQueries, mxSINGLE_CLASS, mxREAL);
  plhs[1] = mxCreateNumericMatrix(k, numQueries, mxSINGLE_CLASS, mxREAL);
#else
  plhs[0] = mxCreateDoubleMatrix(k, numQueries, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(k, numQueries, mxREAL);
#endif
  plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
  plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
  plhs[4] = mxCreateDoubleMatrix(1, 1, mxREAL);
  TimeOut = mxGetPr(plhs[2]);
  double* dTime = mxGetPr(plhs[3]);
  double *sTime = mxGetPr(plhs[4]);
  TimeOut[0] = 0;
  dTime[0] = 0;
  sTime[0] = 0;
  knntype* values = (knntype*)mxGetData(plhs[0]);
  knntype* indices = (knntype*)mxGetData(plhs[1]);

  outbuffDist = (knntype*)malloc(numStreams*CorpusBlocks*maxQueries*sizeofHeap*sizeof(knntype));
  outbuffIdx = (knntype*)malloc(numStreams*CorpusBlocks*maxQueries*sizeofHeap*sizeof(knntype));
  outbuffDist2 = (knntype*)malloc(numStreams*CorpusBlocks*maxQueries*sizeofHeap*sizeof(knntype));
  outbuffIdx2 = (knntype*)malloc(numStreams*CorpusBlocks*maxQueries*sizeofHeap*sizeof(knntype));


  CUstream *stream = (CUstream*)malloc(numStreams*sizeof(CUstream));

  for(int i=0; i<numStreams; i++){
    cuStreamCreate(&stream[i], 0);
  }


  cudaMalloc((void**)&d_data, maxObjects*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotp, maxObjects*sizeof(knntype));
  cudaMalloc((void**)&d_dist, maxObjects*maxQueries*sizeof(knntype));
  cudaMalloc((void**)&d_labels, numStreams*sizeofHeap*maxQueries*sizeof(knntype));
  cudaMalloc((void**)&d_heap, numStreams*sizeofHeap*maxQueries*sizeof(knntype));

  for(int qq=0; qq<numQueries; qq+=MAXQUERYBLOCK){

    cudaEventRecord(start, 0);

    int queryBlockSize = min(MAXQUERYBLOCK, numQueries-qq);
    
    cudaMemcpyAsync(d_query, query+qq*attributes, queryBlockSize*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[0]);
    
    dstfunc.dotP.dotQ<<<queryBlockSize, dstfunc.dotP.nthreadsQ, dstfunc.dotP.externShared*sizeof(knntype), stream[0]>>>(d_dotB, d_query, attributes);
    
    //cudaEventRecord(start, 0);
    
    int count = 0;
    for(int ii=0, c=0; ii<objects; ii+=maxObjects, c++){
      
      int CorpusBlockSize = min(maxObjects, objects-ii);
      int StreamSize = CorpusBlockSize / numStreams;
      
      for(int jj=0; jj<numStreams; jj++){
	cudaMemcpyAsync(d_data + jj*StreamSize*attributes, data + ii*attributes + jj*StreamSize*attributes, StreamSize*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[jj]);
      }
      
      
      for(int jj=0; jj<numStreams; jj++){
	cuknnsHeap(d_dist+jj*StreamSize*queryBlockSize, d_data+jj*StreamSize*attributes, d_query, d_heap+jj*sizeofHeap*queryBlockSize, d_labels+jj*sizeofHeap*queryBlockSize, d_dotp+jj*StreamSize, StreamSize, attributes, queryBlockSize, qk, d_dotB, handle, stream[jj], &TimesOut, ii + jj*StreamSize, &dstfunc);    
      }
      
      for(int jj=0; jj<numStreams; jj++){
	cudaMemcpyAsync(outbuffDist + jj*k*queryBlockSize + c*numStreams*k*queryBlockSize, d_heap + jj*sizeofHeap*queryBlockSize, k*queryBlockSize*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
	cudaMemcpyAsync(outbuffIdx + jj*k*queryBlockSize + c*numStreams*k*queryBlockSize, d_labels + jj*sizeofHeap*queryBlockSize, k*queryBlockSize*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
      }
      count++;
    }
        
    cuCtxSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    TimesOut.knn_time = elapsedTime;
    dTime[0] += TimesOut.dst_time / 1000;
    sTime[0] += TimesOut.srch_time / 1000;
    TimeOut[0] += TimesOut.knn_time / 1000;
    
    
    gettimeofday(&startwtime, NULL);

        
    for(int str=0; str<CorpusBlocks*numStreams; str++){
      transpose_naive(outbuffDist2+str*k*queryBlockSize, outbuffDist+str*k*queryBlockSize, k, queryBlockSize);
      transpose_naive(outbuffIdx2+str*k*queryBlockSize, outbuffIdx+str*k*queryBlockSize, k, queryBlockSize);
    }
    
    gettimeofday(&endwtime, NULL);


    transpose_time += (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

    gettimeofday(&startwtime, NULL);
  
          
    if(CorpusBlocks*numStreams>1){
      
      int qth = queryBlockSize / NTHREADS;
      int qm = queryBlockSize & (NTHREADS-1);
      
      
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
      
      for(int t=0; t<NTHREADS; t++){
	argm[t].data =  outbuffDist2 + t*qth*k;
	argm[t].idx = outbuffIdx2 + t*qth*k;
	argm[t].k = k;
	argm[t].qth = qth + (t == (NTHREADS-1)) * qm;
	argm[t].Q = queryBlockSize;
	argm[t].numStreams = numStreams*CorpusBlocks;
	
	rc = pthread_create(&cputhreads[t], &attr, pmergeRes, (void *)&argm[t]);
	
      }
      
      pthread_attr_destroy(&attr);
      for(int t=0; t<NTHREADS; t++){
	rc = pthread_join(cputhreads[t], &status);
      }
    
    }
    
    gettimeofday(&endwtime, NULL);

    memcpy(values + qq*k, outbuffDist2, queryBlockSize*k*sizeof(knntype));
    memcpy(indices + qq*k, outbuffIdx2, queryBlockSize*k*sizeof(knntype));
    
    //gettimeofday(&endwtime, NULL);

    serial_time += (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

  } 

  //mexPrintf("Serial Parts: %fs\n", serial_time);
  
  //mexPrintf("Transpositon time: %f\n", transpose_time);

  for(int i=0; i<numStreams; i++){
    cuStreamDestroy(stream[i]);
  }

  /*===== clean =======*/
  //free(cputhreads);
  //free(argm);
  free(outbuffDist);
  free(outbuffIdx);
  free(outbuffDist2);
  free(outbuffIdx2);
  cudaFree(d_data);
  cudaFree(d_dotp);
  cudaFree(d_query);
  cudaFree(d_dist);
  cudaFree(d_labels);
  cudaFree(d_heap);
  cudaFree(d_dotB);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaDeviceReset();
}

