NVCC=nvcc
CC=nvcc
NVCCFLAGS=-use_fast_math -O3
CFLAGS=-O3

CUDASES: CudaSurf.o cpdb/cpdb.o cpdb/utils.o SmoothMesh.o
	$(CC) -o CUDASES $(CFLAGS) cpdb/cpdb.o cpdb/utils.o SmoothMesh.o CudaSurf.o


CudaSurf.o: CudaSurf.cu
	$(NVCC) -c -o CudaSurf.o CudaSurf.cu -Icpdb/ $(NVCCFLAGS)

SmoothMesh.o: SmoothMesh.cpp
	$(CC) -c -o SmoothMesh.o SmoothMesh.cpp $(CFLAGS)

cpdb/utils.o: cpdb/utils.cpp
	$(CC) -c -o cpdb/utils.o cpdb/utils.cpp $(CFLAGS)

cpdb/cpdb.o: cpdb/cpdb.cpp
	$(CC) -c -o cpdb/cpdb.o cpdb/cpdb.cpp $(CFLAGS)



clean:
	rm -f *.o CUDASES
	rm -f cpdb/*.o


#nvcc --shared CudaSurf.cu cpdb/*.cpp -Icpdb/ SmoothMesh.cpp -o SESCuda.dll -use_fast_math -O3 --compiler-options "-fPIC"
