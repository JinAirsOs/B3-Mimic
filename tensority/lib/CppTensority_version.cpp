// CppTensority-version.cpp: 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "stdio.h"


#include <iostream>
#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <cuda.h>
#include <helper_cuda.h>
#include "cSimdTs.h"
#include "BytomPoW.h"
#include "seed.h"

#include "CppTensority_version.h"

using namespace std;

BytomMatList* matList_int8;
BytomMatListGpu* matListGpu_int8;
uint8_t result[32] = { 0 };
map <vector<uint8_t>, BytomMatListGpu*> seedCache;
typedef vector<uint8_t> Seed;
typedef vector<Seed> SeedList;
SeedList seed_list;
typedef std::vector<char *> MYARRAY;
static const int cacheSize = 16; //"Answer to the Ultimate Question of Life, the Universe, and Everything"
mutex mtx;


int *SimdTs2(unsigned char blockheader[32], unsigned char seed[32], unsigned char res[32]) {

	mtx.lock();
	Seed seedVec(seed, seed + 32);
	clock_t start, end;

	memset(res, 0, sizeof(res));
	unsigned  char  *result = 0;
	result = res;	

	//cout << "res :" << res << endl;
	//cout << "result : " << result << endl;

	if (seedCache.find(seedVec) != seedCache.end()) {
		//sprintf("\t---%s---\n", "Seed already exists in the cache.");
		matListGpu_int8 = seedCache[seedVec];
	}
	else {
		uint32_t exted[32];
		extend(exted, seed); // extends seed to exted
		Words32 extSeed;
		init_seed(extSeed, exted);

		matList_int8 = new BytomMatList;
		matList_int8->init(extSeed);

		matListGpu_int8 = new BytomMatListGpu;
		initMatVecGpu(matListGpu_int8, matList_int8);
		seedCache.insert(make_pair(seedVec, matListGpu_int8));

		delete matList_int8;
	}

	start = clock();
	iter_mineBytom(blockheader, 32, result);
	end = clock();
	//cout << "iter_mineBytom res :" << res << endl;
	//cout << "item_mineBytom result :" << result << endl;
	//std::cout << "mineBytom: "
	//	<< (double)(end - start) / CLOCKS_PER_SEC << "s"
	//	<< std::endl;

	if (seedCache.size() > cacheSize) {
		seedVec = seed_list.at(cacheSize - 1);
		seed_list.erase(seed_list.begin() + cacheSize - 1);
		seedCache.erase(seedVec);
	}

	mtx.unlock();
	return 0;
}

int InitCUDA()
{
	//return cuInit(0);
	if (cuInit(0) != CUDA_SUCCESS)
	{
		cout << "Init CUDA driver failed" << endl;
		return -1;
	}
	return 0;
}




driverVersionInfo * GetDeviceDriverVersion(int deviceCount, driverVersionInfo *p_driverVersion)
{
	//driverVersionInfo *p_driverVersion = new driverVersionInfo[deviceCount];
	for (int dev = 0; dev < deviceCount; ++dev)
	{
		int driverVersion = 0;
		
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;

		cudaGetDeviceProperties(&deviceProp, dev);
		cudaDriverGetVersion(&driverVersion);

		p_driverVersion[dev].deviceName = deviceProp.name;
		p_driverVersion[dev].deviceVersion = driverVersion;
	}

	return p_driverVersion;
}

int GetDeviceCount()
{

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess)
	{
		cout << "cuDeviceGetCount returned " << (int)error_id << endl << "-> " << cudaGetErrorString(error_id) << endl;
	}

	if (deviceCount == 0)
	{
		cout << "There are no available device(s) that support CUDA" << endl;
	}
	else
	{
		cout << "Detected " << deviceCount << " CUDA Capable device(s)" << endl;
	}
	
	return deviceCount;
}


