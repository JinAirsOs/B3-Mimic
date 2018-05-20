#include <string.h>
#include <time.h>
//#include <sys/time.h>
#include <typeinfo>
#include <cublas.h>
#include <cuda_runtime.h>
#include <string.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "matrix.h"



void test_gpu1()
{
    int n = 256 ;
    char *a = (char *)calloc(n * n, sizeof(char));
    char *b = (char *)calloc(n * n, sizeof(char));
    for (int i = 0; i < n ; i ++){
        for (int j = 0; j < n; j ++)
        {
            a[i * n + j] = (char)(1 + i);
            b[i * n + j] = (char)(1 + j);
        }
    }
    int *c = (int*)calloc(n * n, sizeof(int));
    cucblas_ggemm(a, b ,c);
    for (int i= 0; i < 4; i ++){
        printf("%f \n", (float)c[i]);
    }
}

void test_gpu2()
{
    int n = 256;
    float *a = (float *)calloc(n * n, sizeof(float));
    float *b = (float *)calloc(n * n, sizeof(float));
    float *c = (float *)calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i ++){
        for (int j = 0; j < n; j ++)
        {
            a[i * n + j] = (1 + i);
            b[i * n + j] = (1 + j);
        }
    }
    cucblas_sgemm(a, b ,c);
    for (int i= 0; i < 4; i ++){
        printf("%f \n", c[i]);
    }
}

void cucblas_ggemm(char *h_ma, char *h_mb, int *h_mc) {


    int n = 256;
    int k = 256;
    int m = 256;
    char *mat1_g;
    char *mat2_g;
    int *mat3_g;
    int alpha = 1;
    int beta = 0;
    
    int devID = 0;
    cudaSetDevice(devID);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devID);
    printf("Device : %s, compute SM %d.%d.\n",devProp.name, devProp.major, devProp.minor);
    cudaMalloc((void **)&(mat1_g), sizeof(char) * m * k);
    cudaMalloc((void **)&(mat2_g), sizeof(char) * n * k);
    cudaMalloc((void **)&(mat3_g), sizeof(int) * m * n);
    
    cudaMemcpy(mat1_g, h_ma, sizeof(char) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_g, h_mb, sizeof(char) * n * n, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, mat1_g, CUDA_R_8I, n, mat2_g, CUDA_R_8I, n, &beta, mat3_g, CUDA_R_32I, n, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
    cudaDeviceSynchronize();
    cudaMemcpy(h_mc, mat3_g, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
    cudaFree(mat1_g);
    cudaFree(mat2_g);
    cudaFree(mat3_g);
}

void cucblas_sgemm(float *h_ma, float *h_mb, float *h_mc) {


    int n = 256;
    int k = 256;
    int m = 256;
    cublasOperation_t opt1 = CUBLAS_OP_T;
    cublasOperation_t opt2 = CUBLAS_OP_T;

    float *mat1_g;
    float *mat2_g;
    float *mat3_g;
    float alpha = 1.0;
    float beta = 0.0;
    
    cudaMalloc((void **)&(mat1_g), sizeof(float) * m * k);
    cudaMalloc((void **)&(mat2_g), sizeof(float) * n * k);
    cudaMalloc((void **)&(mat3_g), sizeof(float) * m * n);
    
    cudaMemcpy(mat1_g, h_ma, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_g, h_mb, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cublasSgemm(handle, opt1, opt2, m, n, k, &alpha, mat1_g, n, mat2_g, n, &beta, mat3_g, n);
    //cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, mat1_g, CUDA_R_16F, n, mat2_g, CUDA_R_16F, n, &beta, mat3_g, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
    cudaDeviceSynchronize();
    cudaMemcpy(h_mc, mat3_g, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    cudaFree(mat1_g);
    cudaFree(mat2_g);
    cudaFree(mat3_g);
}
