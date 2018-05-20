#include <cuda_runtime.h>
#include <string.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <time.h>
//#include <sys/time.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace std;


void cucblas_sgemm(float *h_ma, float *h_mb, float *h_mc);
void cucblas_ggemm(char *h_ma, char *h_mb, int *h_mc);
void test_gpu1();
void test_gpu2();
