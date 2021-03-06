#ifndef BYTOMPOW_H
#define BYTOMPOW_H

#include "scrypt.h"
#include "sha3.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include <stdint.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define FNV(v1,v2) int32_t( ((v1)*FNV_PRIME) ^ (v2) )
const int FNV_PRIME = 0x01000193;

void cucblas_dgemm(float *h_ma, float *h_mb, float *h_mc);

struct Mat256x256i8 {
    int8_t d[256][256];

    void toIdentityMatrix() {
        for(int i=0; i<256; i++) {
            for(int j=0; j<256; j++) {
                d[j][i]=0;
            }
        }
        for(int i=0; i<256; i++) {
            d[i][i]=1;
        }
    }

    void copyFrom(const Mat256x256i8& other) {
        for(int i=0; i<256; i++) {
            for(int j=0; j<256; j++) {
                this->d[j][i]=other.d[j][i];
            }
        }
    }

    Mat256x256i8() {
        this->toIdentityMatrix();
    }

    Mat256x256i8(const Mat256x256i8& other) {
        this->copyFrom(other);
    }

    void copyFrom_helper(LTCMemory& ltcMem, int offset) {
        for(int i=0; i<256; i++) {
            const Words32& lo=ltcMem.get(i*4+offset);
            const Words32& hi=ltcMem.get(i*4+2+offset);
            for(int j=0; j<64; j++) {
                uint32_t i32=j>=32?hi.get(j-32):lo.get(j);
                d[j*4+0][i]=(i32>>0)&0xFF;
                d[j*4+1][i]=(i32>>8)&0xFF;
                d[j*4+2][i]=(i32>>16)&0xFF;
                d[j*4+3][i]=(i32>>24)&0xFF;
            }
        }
    }

    void copyFromEven(LTCMemory& ltcMem) {
        copyFrom_helper(ltcMem, 0);
    }

    void copyFromOdd(LTCMemory& ltcMem) {
        copyFrom_helper(ltcMem, 1);
    }
};

struct Arr256x64i32 {
    uint32_t d[256][64];

    uint8_t* d0RawPtr() {
        return (uint8_t*)(d[0]);
    }

    Arr256x64i32(const Mat256x256i8& mat) {
        for(int j=0; j<256; j++) {
            for(int i=0; i<64; i++) {
                d[j][i] = ((uint32_t(uint8_t(mat.d[j][i  + 192]))) << 24) |
                          ((uint32_t(uint8_t(mat.d[j][i + 128]))) << 16) |
                          ((uint32_t(uint8_t(mat.d[j][i  + 64]))) << 8) |
                          ((uint32_t(uint8_t(mat.d[j][i ]))) << 0);
            }
        }
    }

    void reduceFNV() {
        for(int k=256; k>1; k=k/2) {
            for(int j=0; j<k/2; j++) {
                for(int i=0; i<64; i++) {
                    d[j][i] = FNV(d[j][i], d[j + k / 2][i]);
                }
            }
        }
    }
};

struct BytomMatList {
    std::vector<Mat256x256i8*> matVec;

    Mat256x256i8 at(int i) {
        return *(matVec[i]);
    }

    BytomMatList() {
        for(int i=0; i<256; i++) {
            Mat256x256i8* ptr = new Mat256x256i8;
            assert(ptr!=NULL);
            matVec.push_back(ptr);
        }
    }

    ~BytomMatList() {
        for(int i=0; i<256; i++) {
            delete matVec[i];
        }
    }

    void init(const Words32& X_in) {
        Words32 X = X_in;
        LTCMemory ltcMem;
        for(int i=0; i<128; i++) {
            ltcMem.scrypt(X);
            matVec[2*i]->copyFromEven(ltcMem);
            matVec[2*i+1]->copyFromOdd(ltcMem);
        }
    }
};
struct BytomMatListGpu {
  std::vector<int8_t*> matVecGpu;
  int8_t* at(int i) {
    return matVecGpu[i];
  }
  BytomMatListGpu() {
    for(int i=0; i<256; i++) {
      int8_t* devPtr_i8;
      cudaMalloc ((void**)&devPtr_i8, 256*256*sizeof(*devPtr_i8));
      assert(devPtr_i8!=NULL);
      matVecGpu.push_back(devPtr_i8);
    }
  }
  ~BytomMatListGpu() {
    for(int i=0; i<256; i++)
      cudaFree(matVecGpu[i]);
  }
};


extern BytomMatList* matList_int8;
extern BytomMatListGpu* matListGpu_int8;

void mulmatrix( Mat256x256i8 *mat, BytomMatListGpu* matListGpu_int8, std::vector<uint8_t> fourSeq[4]);
void initMatVecGpu(BytomMatListGpu* matListGpu_int8, BytomMatList* matList_int8);

inline void iter_mineBytom(
                        const uint8_t *fixedMessage,
                        uint32_t len,
                        uint8_t result[32]) {
    Mat256x256i8 *mat=new Mat256x256i8;
    sha3_ctx *ctx = (sha3_ctx*)calloc(1, sizeof(*ctx));

    uint8_t sequence[32];
    std::vector<uint8_t> fourSeq[4];

    for(int k=0; k<4; k++) { // The k-loop
        rhash_sha3_256_init(ctx);
        rhash_sha3_update(ctx, fixedMessage+(len*k/4),len/4);//分四轮消耗掉fixedMessage
        rhash_sha3_final(ctx, sequence);
        for(int i=0; i<32; i++){
            fourSeq[k].push_back(sequence[i]);
        }
    }

    mulmatrix(mat, matListGpu_int8, fourSeq);


    Arr256x64i32 arr(*mat);
    arr.reduceFNV();
    rhash_sha3_256_init(ctx);
    rhash_sha3_update(ctx, arr.d0RawPtr(), 256);
    rhash_sha3_final(ctx, result);
    delete mat;
    free(ctx);
}
#endif

