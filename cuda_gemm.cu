#include <iostream>
#include <iomanip>
#include "mkl.h"
#include "utils.h"

#define SINGLE_PRECISION //Comment out to use double precision arithmetic
#define DOUBLE_PRECISION

#ifdef SINGLE_PRECISION
	#define elem_t float
	#define blasGemm cblas_sgemm 
	#define cublasGemm cublasSgemm
#elif defined(DOUBLE_PRECISION)
	#define elem_t double
	#define blasGemm cblas_dgemm 
	#define cublasGemm cublasDgemm
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

#ifndef WARMUPS
#define WARMUPS 3
#endif
#ifndef ITERS
#define ITERS 10
#endif

 void gemmV1(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//write simple kernel
}

#define V2_TILE_M 32
#define V2_TILE_N 32
#define V2_TILE_K 16
 void gemmV2(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//write kernel with shared memory
}

#define V3_TILE_M 64
#define V3_TILE_N 64
#define V3_TILE_K 8
#define V3_THREAD_M 2
#define V3_THREAD_N 2
 void gemmV3(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//write kernel with shared memory and higher arithmetic intensity
}

void runGemmV1(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//call the gpu kernel
}
void runGemmV2(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//call the gpu kernel
}
void runGemmV3(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//call the gpu kernel
}

int main(int argc, char **argv)
{
	float *times = new float[2*ITERS];
	float *timesCPU = times;
	float *timesGPU = times + ITERS;

	elem_t *A, *B, *C, *Cgpu;
	int M = GEMM_M;
	int N = GEMM_N;
	int K = GEMM_K;
	allocateMatrixCPU(M,K,&A);
	allocateMatrixCPU(K,N,&B);
	allocateMatrixCPU(M,N,&C);

	initMatrixRandomCPU<elem_t>(M,K,A);
	initMatrixRandomCPU<elem_t>(K,N,B);
	initMatrixCPU<elem_t>(M,N,C,0.0);

	elem_t *d_A, *d_B, *d_C;
	allocateMatrixGPU(M,K,&d_A);
	allocateMatrixGPU(K,N,&d_B);
	allocateMatrixGPU(M,N,&d_C);

	cudaMemcpy(d_A, A, sizeof(elem_t)*M*K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(elem_t)*N*K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, sizeof(elem_t)*M*N, cudaMemcpyHostToDevice);

	elem_t alpha = 1.0;
	elem_t beta = 0.0;

	//CPU
	struct timespec cpu_start, cpu_end;
	for (int i=0; i<ITERS; i++)
	{
		clock_gettime(CLOCK_MONOTONIC, &cpu_start);
		blasGemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
		clock_gettime(CLOCK_MONOTONIC, &cpu_end);
		timesCPU[i] = computeCPUTime(&cpu_start, &cpu_end);
	}

	//GPU
	for (int i=0; i<WARMUPS; i++)
	{
		runGemmV1(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
		//runGemmV2(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
		//runGemmV3(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
		cudaDeviceSynchronize();
	}
	cudaEvent_t gpu_start, gpu_end;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_end);
	for (int i=0; i<ITERS; i++)
	{
		cudaEventRecord(gpu_start);
		runGemmV1(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
		//runGemmV2(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
		//runGemmV3(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
		cudaEventRecord(gpu_end);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&(timesGPU[i]), gpu_start, gpu_end);
	}
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_end);

	float flops = 2*(float)M*(float)N*(float)K;

	float avg_cpu=0.0;
	for (int i=0; i<ITERS; i++)
		avg_cpu += timesCPU[i];
	avg_cpu = avg_cpu / (float)ITERS;
	std::cout << "==== CPU ====\n";
	std::cout << "Execution time: " << avg_cpu << " ms.\n";
	std::cout << "Performance: " << (flops/1.0e9)/(avg_cpu/1.0e3) << " GFLOP/s.\n";

	float avg_gpu=0.0;
	for (int i=0; i<ITERS; i++)
		avg_gpu += timesGPU[i];
	avg_gpu = avg_gpu / (float)ITERS;
	std::cout << "==== GPU ====\n";
	std::cout << "Execution time: " << avg_gpu << " ms.\n";
	std::cout << "Performance: " << (flops/1.0e9)/(avg_gpu/1.0e3) << " GFLOP/s.\n";

	allocateMatrixCPU(M,N,&Cgpu);
	cudaMemcpy(Cgpu, d_C, sizeof(elem_t)*M*N, cudaMemcpyDeviceToHost);
	std::cout << std::setprecision(10);
	compareMatrices(M,N,C,Cgpu);
	freeMatrixCPU(M,N,Cgpu);

	freeMatrixGPU(M,K,d_A);
	freeMatrixGPU(K,N,d_B);
	freeMatrixGPU(M,N,d_C);

	freeMatrixCPU(M,K,A);
	freeMatrixCPU(K,N,B);
	freeMatrixCPU(M,N,C);

	delete[] times;

}
