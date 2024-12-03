#include <iostream>
#include <iomanip>
#include "mkl.h"
#include "cublas_v2.h"
#include "utils.h"

#define SINGLE_PRECISION //Comment out to use double precision arithmetic
#define DOUBLE_PRECISION

#ifdef SINGLE_PRECISION
	#define elem_t float
	#define blasGemm cblas_sgemm 
	#define cublasGemm cublasSgemm
	#define cublasGemmBatched cublasSgemmBatched
#elif defined(DOUBLE_PRECISION)
	#define elem_t double
	#define blasGemm cblas_dgemm 
	#define cublasGemm cublasDgemm
	#define cublasGemmBatched cublasDgemmBatched
#endif

#ifndef GEMM_M
#define GEMM_M 256
#endif
#ifndef GEMM_N
#define GEMM_N 256
#endif
#ifndef GEMM_K
#define GEMM_K 256
#endif

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef NB_STREAMS
#define NB_STREAMS 16
#endif

#ifndef WARMUPS
#define WARMUPS 1
#endif
#ifndef ITERS
#define ITERS 10
#endif

void tileGemm(cublasHandle_t handle, int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC, int tileM, int tileN)
{
	//TODO: TASK 3
}

void tileGemmStreams(cublasHandle_t handle, int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC, int tileM, int tileN, int nb_streams, cudaStream_t *streams)
{
	//TODO: TASK 4
}

void tileGemmBatch(cublasHandle_t handle, int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC, int tileM, int tileN)
{
	//TODO: TASK 5
}

int main(int argc, char **argv)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaStream_t *streams;
       	createStreams(NB_STREAMS, &streams);

	float *times = new float[2*ITERS];
	float *timesCPU = times;
	float *timesGPU = times + ITERS;

	elem_t *A, *B, *C, *Cgpu;
	elem_t *d_A, *d_B, *d_C;
	int M = GEMM_M;
	int N = GEMM_N;
	int K = GEMM_K;

	//TODO: TASK 1 (Allocate and init A,B,C)

	//TODO: TASK 2.1 (Allocate and init d_A, d_B, d_C)

	elem_t alpha = 1.0;
	elem_t beta = 0.0;

	//CPU
	struct timespec cpu_start, cpu_end;
	for (int i=0; i<ITERS; i++)
	{
		clock_gettime(CLOCK_MONOTONIC, &cpu_start);
		//TODO: TASK 1 (Run blasGemm)
		clock_gettime(CLOCK_MONOTONIC, &cpu_end);
		timesCPU[i] = computeCPUTime(&cpu_start, &cpu_end);
	}

	//GPU
	for (int i=0; i<WARMUPS; i++)
	{
		//TODO: TASK 2.2 (run cublasGemm)
		//TODO: TASK 3
		//TODO: TASK 4
		//TODO: TASK 5
		cudaDeviceSynchronize();
	}
	cudaEvent_t gpu_start, gpu_end;
	for (int i=0; i<ITERS; i++)
	{
		//TODO: TASK 2.2 (run cublasGemm)
		//TODO: TASK 3
		//TODO: TASK 4
		//TODO: TASK 5
		//TODO: TASK 2.2 (Measure execution times)
		cudaDeviceSynchronize();
	}

	//TODO: TASK 1 (Compute and print average execution time on CPU)

	//TODO: TASK 2.2 (Compute and print average execution time on GPU)

	//TODO: TASK 2.2 (Compare CPU and GPU output)

	//TODO: TASK 2.1 (Free d_A, d_B, d_C)

	//TODO: TASK 1 (Free A,B,C)

	destroyStreams(NB_STREAMS, streams);
	cublasDestroy(handle);

	delete[] times;

}
