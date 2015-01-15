#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cublas.h"
#include "../include/cuknns.h"
#include "../include/shared_functions.h"

#define DIMENTIONS 128
//#define MAXQUERYBLOCK 65536
//#define MAXQUERYBLOCK 32768
#define MAXQUERYBLOCK 128

char* errMsg = "Only 128 dimentional data curentrly suported\n";

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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  /*=========MATLAB pointers=========*/
  knntype *data;//, *dp;
  knntype *query;
  knntimes TimesOut;
  size_t memory_free, memory_total;
  double *knnTime, *dstTime, *srchTime;

  TimesOut.dst_time = 0;
  TimesOut.srch_time = 0;
  TimesOut.knn_time = 0;

  /*========= Devece pointers ======*/
  knntype *d_data, *d_query, *d_labels, *d_dist, *d_dotp;

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

  int qk = (int)(log((float)k) / log(2.0));

  plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
  plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
  plhs[4] = mxCreateDoubleMatrix(1, 1, mxREAL);
  knnTime = mxGetPr(plhs[2]);
  dstTime = mxGetPr(plhs[3]);
  srchTime = mxGetPr(plhs[4]);
  knnTime[0] = 0;
  dstTime[0] = 0;
  srchTime[0] = 0;

  knntype *d_dotB;

  int maxQueries = min(numQueries, MAXQUERYBLOCK);

  cudaMalloc((void**)&d_query, maxQueries*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotB, maxQueries*sizeof(knntype));


  /* find maximum corpus block that can be processed */
  cuMemGetInfo(&memory_free, &memory_total);
  int memory_req = (4*maxQueries + attributes + 1)*sizeof(knntype);

  int maxObjects = (int)ceil((float)memory_free*0.9 / (float)memory_req);

  maxObjects = (1 << (int)floor(log((double)maxObjects)/log(2)));

  int reqStreams = 1;

  int CorpusBlocks = (int)ceil((float)objects/(float)maxObjects);

  numStreams = max(numStreams, reqStreams);

  maxObjects = min(maxObjects, objects);

  /* initialize distance functions */
  distFunctParam dstfunc;
  dstfunc.distF = pdist_NT;
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


  /*===========Initialize the output===================*/
#ifndef __DOUBLE__
  plhs[0] = mxCreateNumericMatrix(k, numQueries, mxSINGLE_CLASS, mxREAL);
  plhs[1] = mxCreateNumericMatrix(k, numQueries, mxSINGLE_CLASS, mxREAL);
#else
  plhs[0] = mxCreateDoubleMatrix(k, numQueries, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(k, numQueries, mxREAL);
#endif
  knntype* values = (knntype*)mxGetData(plhs[0]);
  knntype* indices = (knntype*)mxGetData(plhs[1]);

  knntype* outbuff_dist = (knntype*)malloc(k*numStreams*CorpusBlocks*maxQueries*sizeof(knntype));
  knntype* outbuff_idx = (knntype*)malloc(k*numStreams*CorpusBlocks*maxQueries*sizeof(knntype));

  CUstream *stream = (CUstream*)malloc(numStreams*sizeof(CUstream));

  for(int i=0; i<numStreams; i++){
    cuStreamCreate(&stream[i], 0);
  }

  /*Memory allocation*/
  knntype *distbuff, *idxbuff;
  cudaMalloc((void**)&d_data, maxObjects*attributes*sizeof(knntype));
  cudaMalloc((void**)&d_dotp, maxObjects*sizeof(knntype));
  cudaMalloc((void**)&d_dist, maxObjects*maxQueries*sizeof(knntype));
  cudaMalloc((void**)&d_labels, maxObjects*maxQueries*sizeof(knntype));
  cudaMalloc((void**)&distbuff, maxQueries*maxObjects*sizeof(knntype));
  cudaMalloc((void**)&idxbuff, maxQueries*maxObjects*sizeof(knntype));


  for(int qq=0; qq<numQueries; qq+=MAXQUERYBLOCK){

    int queryBlockSize = min(MAXQUERYBLOCK, numQueries-qq);
    
    cudaMemcpyAsync(d_query, query+qq*attributes, queryBlockSize*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[0]);
    
    /*compute the dot product of the queries*/
    dstfunc.dotP.dotQ<<<queryBlockSize, dstfunc.dotP.nthreadsQ, dstfunc.dotP.externShared*sizeof(knntype), stream[0]>>>(d_dotB, d_query, attributes);
    
    cudaEventRecord(start, 0);
    
    for(int ii=0, c = 0; ii<objects; ii+=maxObjects, c++){
      
      int CorpusBlockSize = min(maxObjects, objects-ii);
      int StreamSize = CorpusBlockSize / numStreams;
      for(int jj=0; jj<numStreams; jj++){
	cudaMemcpyAsync(d_data + jj*StreamSize*attributes, data + ii*attributes + jj*StreamSize*attributes, StreamSize*attributes*sizeof(knntype), cudaMemcpyHostToDevice, stream[jj]);
      }
      
      for(int jj=0; jj<numStreams; jj++){
	cuknnsBitonicSTR(d_dist + jj*StreamSize*queryBlockSize, d_data + jj*StreamSize*attributes, d_query, d_labels + jj*StreamSize*queryBlockSize, d_dotp+jj*StreamSize, d_dotB, distbuff + jj*StreamSize*queryBlockSize, idxbuff + jj*StreamSize*queryBlockSize, StreamSize, attributes, queryBlockSize, k, handle, stream[jj], &TimesOut, c*numStreams + jj, &dstfunc);
      }    
      
      for(int jj=0; jj<numStreams; jj++){
	cudaMemcpyAsync(outbuff_dist + jj*k*queryBlockSize + c*numStreams*k*queryBlockSize, d_dist + jj*StreamSize*queryBlockSize, k*queryBlockSize*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
	cudaMemcpyAsync(outbuff_idx + jj*k*queryBlockSize + c*numStreams*k*queryBlockSize, d_labels + jj*StreamSize*queryBlockSize, k*queryBlockSize*sizeof(knntype), cudaMemcpyDeviceToHost, stream[jj]);
      }

    }
    
    cuCtxSynchronize();
      
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    TimesOut.knn_time += elapsedTime / 1000;
    knnTime[0] += TimesOut.knn_time;
    dstTime[0] += TimesOut.dst_time / 1000;
    srchTime[0] += TimesOut.srch_time / 1000;
   
    
    if(CorpusBlocks*numStreams>1){
      mergeRes(outbuff_dist, outbuff_idx, k, queryBlockSize, numStreams*CorpusBlocks);
    }
   
    
    memcpy(values + qq*k, outbuff_dist, k*queryBlockSize*sizeof(knntype));
    memcpy(indices + qq*k, outbuff_idx, k*queryBlockSize*sizeof(knntype));
   
  }
  
  for(int i=0; i<numStreams; i++){
    cuStreamDestroy(stream[i]);
  }

  free(stream);
  free(outbuff_dist);
  free(outbuff_idx);

  /*======clean======*/
  cublasDestroy(handle);
  cudaFree(d_data);
  cudaFree(d_dotp);
  cudaFree(d_dotB);
  cudaFree(d_query);
  cudaFree(d_dist);
  cudaFree(d_labels);
  cudaFree(distbuff);
  cudaFree(idxbuff);
  cudaDeviceReset();
  /////////
}








