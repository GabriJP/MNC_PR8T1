/*
Ejemplo de uso de cuBLAS a nivel 3 con dgemm()
Comparamos con la versión de MKL de la CPU

Gabriel Jiménez para MNC, gabriel.jimenez102@alu.ulpgc.es
*/

#include <cstdio>
#include <cstring>
#include <random>

#include <mkl.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "eTimer.h"

#define N 2*1024

int main(int arcg, char* argv[])
{
	double *host_A, *host_B, *host_C;
	double *dev_A, *dev_B, *dev_C;
	int sizematrix = N * N * sizeof(double);
	auto alpha = 1.0, beta = 0.0;

	std::random_device gen;
	std::normal_distribution<double> dist(0.0, 1.0);

	host_A = static_cast<double*>(mkl_malloc(sizematrix, 64));
	host_B = static_cast<double*>(mkl_malloc(sizematrix, 64));
	host_C = static_cast<double*>(mkl_malloc(sizematrix, 64));

	for (auto y = 0; y < N; y++)
	{
		for (auto x = 0; x < N; x++)
		{
			host_A[y * N + x] = dist(gen);
			host_B[y * N + x] = dist(gen);
		}
	}

	auto Tcpu = eTimer();
	auto Tgpu = eTimer();

	//Un nuevo uso de eTimer
	// Dado que dgemm en CPU no destrue sus datos y beta=0, podemos repetir el cálculo muchas veces
	//para obtener mínimos, promedio y máximo
	for (auto i = 0; i < 10; i++)
	{
		Tcpu.start();
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, host_A, N, host_B, N, beta, host_C, N);
		Tcpu.stop();
	}
	Tcpu.report("CPU");

	for (auto x = 0; x < 5; x++)
	{
		printf("%g ", host_C[x]);
	}
	printf("\n");

	memset(host_C, 0, sizematrix);

	//Codigo de la GPU
	cudaError_t cudaStatus;
	cublasStatus_t cublasStatus;
	cublasHandle_t handle;

	auto cudaDevice = 1;
	cudaStatus = cudaGetDevice(&cudaDevice);
	cublasStatus = cublasCreate(&handle);

	//Reservar espacio en GPU
	cudaStatus = cudaMalloc(&dev_A, sizematrix);
	cudaStatus = cudaMalloc(&dev_B, sizematrix);
	cudaStatus = cudaMalloc(&dev_C, sizematrix);

	//Transfiere datos al estilo cuda o cublas
	//cudaStatus = cudaMemcpy(dev_A, host_A, sizematrix, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(dev_B, host_B, sizematrix, cudaMemcpyHostToDevice);
	cublasStatus = cublasSetMatrix(N, N, sizeof(double), host_A, N, dev_A, N);
	cublasStatus = cublasSetMatrix(N, N, sizeof(double), host_B, N, dev_B, N);

	Tgpu.start();
	//cublas dgemm. CuBLAS asume codificación Fortran, luesgo debe usar Transpose
	cublasStatus = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, dev_A, N, dev_B, N, &beta, dev_C, N);
	cudaStatus = cudaDeviceSynchronize();
	Tgpu.stop();
	Tgpu.report("GPU");

	//Recupera resultados al estilo cuda o cublas
	//cudaStatus = cudaMemcpy(host_c, dev_c, sizematrix, cudaMemcpy,DeviceToHost);
	cublasStatus = cublasGetMatrix(N, N, sizeof(double), dev_C, N, host_C, N);

	//Pero los datos recuperados están traspuestos
	for (auto i = 0; i < 5; i++)
	{
		printf("%g ", host_C[i * N]);
	}
	printf("\n");

	//Liberación de recursos
	cudaStatus = cudaFree(dev_A);
	cudaStatus = cudaFree(dev_B);
	cudaStatus = cudaFree(dev_C);
	cublasStatus = cublasDestroy(handle);

	cudaStatus = cudaDeviceReset(); //lo último referente a la GPU
	//fin de la GPU

	mkl_free(host_A);
	mkl_free(host_B);
	mkl_free(host_C);

	getchar();
	return 0;
}
