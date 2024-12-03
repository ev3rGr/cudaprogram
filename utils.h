#include <iostream>
#include <random>
#include <time.h>

#define TOL 2e-5

template <typename T>
void allocateMatrixGPU(int M, int N, T** ptr)
{
	//TODO: TASK 2.1
}

template <typename T>
void freeMatrixGPU(int M, int N, T* ptr)
{
	//TODO: TASK 2.1
}


float computeCPUTime(struct timespec *s, struct timespec *e)
{
	return (e->tv_sec-s->tv_sec)*(float)1e3 + (e->tv_nsec-s->tv_nsec)/(float)1e6;
}	

void createStreams(int n, cudaStream_t **streams)
{
	*streams = new cudaStream_t[n];
	for (int i=0; i<n; i++)
		cudaStreamCreate(*streams+i);
}

void destroyStreams(int n, cudaStream_t *streams)
{
	for (int i=0; i<n; i++)
		cudaStreamDestroy(streams[i]);
}

template <typename T>
void allocateMatrixCPU(int M, int N, T** ptr)
{
	*ptr = new T[M*N];
}

template <typename T>
void freeMatrixCPU(int M, int N, T* ptr)
{
	delete[] ptr;
}

template <typename T>
void initMatrixCPU(int M, int N, T* ptr, T val)
{
	for (int i=0; i<M*N; i++)
		ptr[i] = val;
}

template <typename T>
void initMatrixRandomCPU(int M, int N, T *ptr)
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<T> dist(0,1);
	for (int i=0; i<M*N; i++)
		ptr[i] = dist(mt);
}

template <typename T>
T frobenius_norm(int M, int N, T* ptr)
{
	T sum = static_cast<T>(0.0);
	for (int i=0; i<M*N; i++)
	{
		T abs_val = std::abs(ptr[i]);
		sum += abs_val*abs_val;
	}
	return sqrt(sum);
}

template <typename T>
T frobenius_norm_low(int N, T* ptr)
{
	T sum = static_cast<T>(0.0);
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<=i; j++)
		{
			T abs_val = std::abs(ptr[i + j*N]);
			sum += abs_val*abs_val;
		}
	}
	return sqrt(sum);
}

template <typename T>
void compareMatrices(int M, int N, T* ref, T* test)
{
	T tolerance = static_cast<T>(TOL);
	T norm_ref = frobenius_norm(M,N,ref);
	T norm_test = frobenius_norm(M,N,test);
	T diff = std::abs(norm_ref-norm_test);
	std::cout << "Ref has norm " << norm_ref << ".\nTest has norm " << norm_test << ".\nDiff is " << diff << "\n";
	T maxAbsDiff = static_cast<T>(0.0);
	T maxRelDiff = static_cast<T>(0.0);
	for (int i=0; i<M*N; i++)
	{
		T diff = std::abs(ref[i]-test[i]);
		/*if (diff > tolerance)
		{
			std::cerr << "Warning: matrices do not match at position (" << i%M << "," << i/M << ") !\n";
		}*/
		
		if (diff/(diff+ref[i]) > maxRelDiff)
		{
			maxAbsDiff = diff;
			maxRelDiff = diff/(diff+ref[i]);
		}
	}
	std::cout << "Max relative error : " << maxRelDiff << " (abs=" << maxAbsDiff << ").\n";
	if (maxAbsDiff <= tolerance || maxRelDiff <= tolerance)
		std::cout << "CORRECT\n";
	else
		std::cout << "INCORRECT\n";
}

template <typename T>
void compareMatricesLow(int N, T* ref, T* test)
{
	T tolerance = static_cast<T>(TOL*100);
	T norm_ref = frobenius_norm_low(N,ref);
	T norm_test = frobenius_norm_low(N,test);
	T diff = std::abs(norm_ref-norm_test);
	std::cout << "Ref has norm " << norm_ref << ".\nTest has norm " << norm_test << ".\nDiff is " << diff << "\n";
	T maxRelDiff = static_cast<T>(0.0);
	T maxAbsDiff = static_cast<T>(0.0);
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<=i; j++)
		{
			size_t index = i + j*N;
			T diff = std::abs(ref[index]-test[index]);
			/*if (diff > tolerance)
			{
				std::cerr << "Warning: matrices do not match at position (" << i%M << "," << i/M << ") !\n";
			}*/
			if (diff/(diff+ref[index]) > maxRelDiff)
			{
				maxAbsDiff = diff;
				maxRelDiff = diff/(diff+ref[index]);
			}
		}
	}
	std::cout << "Max relative error : " << maxRelDiff << " (abs=" << maxAbsDiff << ").\n";
	if (maxAbsDiff <= tolerance || maxRelDiff <= tolerance)
		std::cout << "CORRECT\n";
	else
		std::cout << "INCORRECT\n";
}

template <typename T>
void printMatrixCPU(int M, int N, T* mat)
{
	for (int i=0; i<M; i++)
	{
		for (int j=0; j<N; j++)
		{
			printf("%.3f ", mat[i + j*M]);
		}
		printf("\n");
	}
}

template <typename T>
__global__ void kernelPrintMatrixGPU(int M, int N, T* mat)
{
	if (threadIdx.x > 0 || blockIdx.x > 0)
		return;

	for (int i=0; i<M; i++)
	{
		for (int j=0; j<N; j++)
		{
			printf("%.3f ", mat[i + j*M]);
		}
		printf("\n");
	}
}

template <typename T>
void printMatrixGPU(int M, int N, T *mat)
{
	kernelPrintMatrixGPU<<<1,1>>>(M,N,mat);
}
