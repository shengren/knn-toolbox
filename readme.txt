The current version V 2.1.0 of GPU KNNS compiles under Linux and requires the
installation of CUDA (version 4.2 of later) and a NVIDIA CUDA enabled
GPU. Also MATLAB (any version) is recommended 

To build successfully, the macros CUDA_PATH and MATLAB_ROOT have to be
modified to contain the installation paths of CUDA and MATLAB in your
computer. Also modify the macro ARCH to contain the number of the GPUs
compute capabilities multiplied by 10. i.e. if your compute capabilities are
1.3 give the value 13.


COMPILING:

To compile the entire library with the C and MATLAB interface issue:

make clean; make

After that the libraries (libcuknn.a and libcuknn_double.a) should 
appear under ./lib and the mex 
functions of the MATLAB interface should appear under ./bin

To compile only the library without the MATLAB interface issue:

make clean; make library

To use as a MATLAB toolbox call the setup script in the main folder.


TEST:

To verify installation run the knnTest script under ./demos from MATLAB. If
the printed messages are "PASS" the library has been successfully installed. 


USE:

For an example of how to use the library in a C program see the /src/knnTest.cu file.

For an example of how to use the MATLAB interface see the /demo/example1.m file

For an example of how to use the streaming feature see the /demo/Testbin.m file

For an example of how to use compile and link when using the library see
the bintest tag at the Makefile 

For an example of how to use the MATLAB interface to solve the all-kNN problem see /demo/allknnTest.m
The performance in that case is not optimal yet.


SIMPLE C FUNCTION REFERENCE:

void knnsplan(knnplan *plan, int N, int Q, int D, int k);

plan: The planer for the execution
N: The number of data points in the corpus
Q: The number of queries
D: The number of dimensions of the corpus and query sets
k: The number of neighbors


void knnsexecute(knnplan plan, float *data, float *queries, float *KNNdist, float *KNNidx);

plan: The planer of the execution
*data: Pointer to the corpus. D must be the leading dimension
*queries: Pointer to the queries. D must be the leading dimension
*KNNdist: Pointer to the distances of the nearest neighbors. k is always
the leading dimension. The matrix is not always ordered 
*KNNidx: Pointer to the indexes of the nearest neighbors. k is always the
leading dimension. The matrix is not always ordered. 

IMPORTANT NOTE:

The library supports:

Any number of data points in the corpus.

All data points in the corpus or query set must be single or double 
precision floating points

The number of neighbors must be a power of two smaller or equal to 512.

The sizes of corpus and query sets are recommended to be to powers of two for
optimal performance.

Please report any issue and features you would like to be included in
future releases at nsismani@auth.gr 



WHAT IS NEW FROM VERSION 1.1.0:

x4 times faster the truncated bitonic sort (TBiS) algorithm

No limit to the size of the corpus that can be processed

Works for arbitrary corpus sizes not only multiples of 512

Double precision floating points are supported

The optimal number of CUDA streams is defined by the planner.



WHAT IS NEW FROM VERSION 2.0.0

Works for any number of dimensions

Works for any number of queries




