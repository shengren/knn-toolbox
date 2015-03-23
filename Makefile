# Makefile
# AUTHOR: Nikos Sismanis
# Date: Mar 2012
ARC=21


MEX=mex -O
NVCC=nvcc -O4 -arch=sm_$(ARC) -Xcompiler -fPIC
CC=gcc -O4
CMP=tar

# Set the correct paths in your computer

CUDA_PATH=/usr/local/cuda-6.5
CUDA_INC=$(CUDA_PATH)/bin/
CUDA_LIB=$(CUDA_PATH)/lib64/
MATLAB_ROOT=/usr/local/MATLAB/R2011b


##############################################
# Do not modify anything beyond that point
#############################################

MEX_INC=$(MATLAB_ROOT)/extern/include
LIBS=-lcuda -lcudart -lcublas

SRC=src
BIN=bin
LIB=lib
INC=include
TOOL=toolbox
DEMO=demo
DATA=data
RELEASE_VERSION=cuknn-toolbox-v2.1.0

AA=../knn-toolbox/$(SRC)
ADDSRC=$(AA)/cuknnBitonic.cu $(AA)/cuknnHeap.cu $(AA)/mexknnsGen.cu $(AA)/mexknnsHeap.cu $(AA)/knnsplan.cu $(AA)/knnsexecute.cu $(AA)/knnTest.cu
ADDINC=../knn-toolbox/$(INC)/*.h
BB=../knn-toolbox/$(TOOL)
ADDTOOL=$(BB)/gpuknnBitonic.m $(BB)/gpuknnHeap.m $(BB)/gpuknns.m $(BB)/knnsplan.m
ADDBIN=../knn-toolbox/$(BIN)/bin.txt
DD = ../knn-toolbox/$(DEMO)
ADDDEMO=$(DD)/exportData.m $(DD)/importData.m $(DD)/example1.m $(DD)/knn.m $(DD)/knnTestDouble.m $(DD)/knnTest.m $(DD)/Testbin.m $(DD)/example2.m $(DD)/allknnTest.m
ADDLIB = ../knn-toolbox/$(LIB)/lib.txt
CC=../knn-toolbox/$(DATA)
ADDDATA = $(CC)/data.txt

all: library bintest mexfiles


library: $(SRC)/cuknnHeap.cu $(SRC)/cuknnBitonic.cu

	$(NVCC) -c -D CUARCH=$(ARC) $(SRC)/cuknnHeap.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/cuknnHeap.o
	$(NVCC) -c -D CUARCH=$(ARC) $(SRC)/cuknnBitonic.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/cuknnBitonic.o
	$(NVCC) -c -D CUARCH=$(ARC) $(SRC)/knnsplan.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/knnsplan.o
	$(NVCC) -c -D CUARCH=$(ARC) $(SRC)/knnsexecute.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/knnsexecute.o
	ar rcs $(LIB)/libcuknn.a $(LIB)/cuknnHeap.o $(LIB)/cuknnBitonic.o $(LIB)/knnsplan.o $(LIB)/knnsexecute.o


	$(NVCC) -c -D CUARCH=$(ARC) -D __DOUBLE__=1 $(SRC)/cuknnHeap.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/cuknnDHeap.o
	$(NVCC) -c -D CUARCH=$(ARC) -D __DOUBLE__=1 $(SRC)/cuknnBitonic.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/cuknnDBitonic.o
	$(NVCC) -c -D CUARCH=$(ARC) -D __DOUBLE__=1 $(SRC)/knnsplan.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/knnsDplan.o
	$(NVCC) -c -D CUARCH=$(ARC) -D __DOUBLE__=1 $(SRC)/knnsexecute.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/knnsDexecute.o
	ar rcs $(LIB)/libcuknn_double.a $(LIB)/cuknnDHeap.o $(LIB)/cuknnDBitonic.o $(LIB)/knnsDplan.o $(LIB)/knnsDexecute.o



bintest: $(SRC)/knnTest.cu

	$(NVCC) -c $(SRC)/knnTest.cu -I$(CUDA_INC) -o $(LIB)/knnTest.o
	$(NVCC) $(LIB)/knnTest.o -L$(LIB) -lcuknn -L$(CUDA_LIB) $(LIBS) -o $(DEMO)/knnTest
	$(NVCC) -D __DOUBLE__=1 $(SRC)/knnTest.cu -L$(LIB) -lcuknn_double -L$(CUDA_LIB) $(LIBS) -o $(DEMO)/knnTestDouble


release:

	$(CMP) -cvf $(RELEASE_VERSION).tar $(ADDSRC) $(ADDINC) $(ADDTOOL) $(ADDDEMO) $(ADDLIB) $(ADDDATA) $(ADDBIN) ../knn-toolbox/Makefile ../knn-toolbox/setup.m ../knn-toolbox/readme.txt

bitonicInter: $(SRC)/cuknnBitonicInter.cu

	$(NVCC) -c $(SRC)/cuknnBitonicInter.cu -I$(CUDA_INC) -o $(LIB)/cuknnBitonicInter.o
	$(NVCC) -c $(SRC)/mexknnsBitonicInter.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/mexknnsBitonicInter.o
	$(MEX) $(LIB)/cuknnBitonicInter.o $(LIB)/mexknnsBitonicInter.o $(LIB)/libcuknn.a -L$(CUDA_LIB) $(LIBS) -o $(BIN)/mexknnsBitonicInter


mexfiles: $(SRC)/mexknnsGen.cu

	$(NVCC) -c -D CUARCH=$(ARC) $(SRC)/mexknnsGen.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/mexknnsBitonic.o
	$(NVCC) -c -D CUARCH=$(ARC) -D __DOUBLE__=1 $(SRC)/mexknnsGen.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/mexknnsDBitonic.o
	$(MEX) $(LIB)/mexknnsBitonic.o $(LIB)/libcuknn.a -L$(CUDA_LIB) $(LIBS) -o $(BIN)/mexknnsBitonic
	$(MEX) $(LIB)/mexknnsDBitonic.o $(LIB)/libcuknn_double.a -L$(CUDA_LIB) $(LIBS) -o $(BIN)/mexknnsDBitonic
	$(NVCC) -c -D ARCH=$(ARC) $(SRC)/mexknnsHeap.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/mexknnsHeap.o
	$(NVCC) -c -D CUARCH=$(ARC) -D __DOUBLE__=1 $(SRC)/mexknnsHeap.cu -I$(CUDA_INC) -I$(MEX_INC) -o $(LIB)/mexknnsDHeap.o
	$(MEX) $(LIB)/mexknnsHeap.o $(LIB)/libcuknn.a -L$(CUDA_LIB) $(LIBS) -o $(BIN)/mexknnsHeap
	$(MEX) $(LIB)/mexknnsDHeap.o $(LIB)/libcuknn_double.a -L$(CUDA_LIB) $(LIBS) -o $(BIN)/mexknnsDHeap

clean:
	rm -f *~
	rm -f $(LIB)/*o $(BIN)/*mexa64
	rm -f $(SRC)/*~
	rm -f $(LIB)/*.a
	rm -f $(TOOL)/*~
	rm -f $(DEMO)/*~
	rm -f *.tar


