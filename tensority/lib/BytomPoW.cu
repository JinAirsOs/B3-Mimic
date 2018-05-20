#include <string.h>
#include <time.h>
//#include <sys/time.h>
#include <typeinfo>
#include <cublas.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "BytomPoW.h"

//__device__ float mat7_g[16][256][256 * 256];

using namespace std;
__global__ void matrix_transpose(int k, int *x, int8_t *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int n = i / k;
    //int m = i - n * k;
    int v = (x[i]);
    //y[m * k + n] = (int8_t)(((v&0xFF)+ ((v>>8)&0xFF))&0xFF);
    y[i] = (int8_t)(((v&0xFF)+ ((v>>8)&0xFF))&0xFF);
}

__global__ void matrix_add(int k, int8_t *x, int8_t *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = x[i] + y[i];
    x[i] = (int8_t)tmp;//(tmp&0xFF);
}


__global__ void initial(int k, int8_t *x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = i / k;
    int m = i - n * k;
    if (m == n)
        x[i] = 1;
    else
        x[i] = 0;
}

void initMatVecGpu(BytomMatListGpu* matListGpu_int8, BytomMatList* matList_int8) {
  for(int i=0; i<matList_int8->matVec.size(); i++) {
    int8_t* hostPtr_i8 = (int8_t*)(matList_int8->at(i).d);
    int8_t* devPtr_i8 = (int8_t*)(matListGpu_int8->at(i));
    cublasStatus_t stat = cublasSetMatrix (256, 256, sizeof(*devPtr_i8), hostPtr_i8, 256, devPtr_i8, 256);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr<<"Fail to Set CuBlas Matrix."<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

void mulmatrix(Mat256x256i8 *mat, BytomMatListGpu* matListGpu_int8, std::vector<uint8_t> fourSeq[4]){

    clock_t start, end;
    start = clock();
    int n = 256;
    int k = 256;
    int m = 256;
    cublasOperation_t opt1 = CUBLAS_OP_N;
    cublasOperation_t opt2 = CUBLAS_OP_N;
    int8_t *a; // a
    int8_t *b; // b
    int *c; // c
    int8_t *mat4_g; // tmp
    int8_t *mat5_g; // sum
    const int alpha = 1;
    const int beta = 0;
    cudaMalloc((void **)&(a), sizeof(int8_t) * m * k);
    cudaMalloc((void **)&(b), sizeof(int8_t) * n * k);
    cudaMalloc((void **)&(c), sizeof(int) * m * n);
    cudaMalloc((void **)&(mat4_g), sizeof(int8_t) * m * n);
    cudaMalloc((void **)&(mat5_g), sizeof(int8_t) * m * n);
    cublasHandle_t handle;
    cublasCreate(&handle);
    int grid = 256;
    int block = 256;
    initial<<<grid, block>>>(n, mat4_g);
    for(int ki=0; ki<4; ki++) { // The k-loop
        for(int j=0; j<2; j++) {
            for(int i=0; i<32; i+=2) {
                if (i + j == 0)
                    cublasGemmEx(handle, opt1, opt2, n, m, k, &alpha, matListGpu_int8->at(fourSeq[ki][i]), CUDA_R_8I, n, mat4_g, CUDA_R_8I, n, &beta, c, CUDA_R_32I, n, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
                else
                    cublasGemmEx(handle, opt1, opt2, n, m, k, &alpha, matListGpu_int8->at(fourSeq[ki][i]), CUDA_R_8I, n, a, CUDA_R_8I, n, &beta, c, CUDA_R_32I, n, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
                matrix_transpose<<<grid, block>>>(n, c, a);
                cublasGemmEx(handle, opt1, opt2, n, m, k, &alpha, matListGpu_int8->at(fourSeq[ki][i+1]), CUDA_R_8I, n, a, CUDA_R_8I, n, &beta, c, CUDA_R_32I, n, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
                matrix_transpose<<<grid, block>>>(n, c, a);
            }
        }
        matrix_add<<<grid, block>>>(n, mat5_g, a);
    }
    cudaMemcpy(mat->d, mat5_g, sizeof(int8_t) * n * n, cudaMemcpyDeviceToHost);
	end = clock();
//printf("%d \n", mat->d[0][0]);
	//std::cout << "mulmatrix: "
	//	<< (double)(end - start) / CLOCKS_PER_SEC << "s"
	//	<< std::endl;
    cublasDestroy(handle);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(mat4_g);
    cudaFree(mat5_g);
}
