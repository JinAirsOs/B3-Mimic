#include <iostream>
#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include "cSimdTs.h"
#include "BytomPoW.h"
#include "seed.h"

using namespace std;

BytomMatList* matList_int8;
BytomMatListGpu* matListGpu_int8;
uint8_t result[32] = {0};
map <vector<uint8_t>, BytomMatListGpu*> seedCache;
typedef vector<uint8_t> Seed;
typedef vector<Seed> SeedList;
SeedList seed_list;
typedef std::vector<char *> MYARRAY;
static const int cacheSize = 16; //"Answer to the Ultimate Question of Life, the Universe, and Everything"
mutex mtx;


uint8_t *SimdTs(unsigned char blockheader[32], unsigned char seed[32]){
    mtx.lock();
    Seed seedVec(seed, seed + 32);
    clock_t start, end;

    if(seedCache.find(seedVec) != seedCache.end()) {
        printf("\t---%s---\n", "Seed already exists in the cache.");
        matListGpu_int8 = seedCache[seedVec];
    } else {
        uint32_t exted[32];
        extend(exted, seed); // extends seed to exted
        Words32 extSeed;
        init_seed(extSeed, exted);

        matList_int8= new BytomMatList;
        matList_int8->init(extSeed);

        matListGpu_int8=new BytomMatListGpu;
        initMatVecGpu(matListGpu_int8, matList_int8);
        seedCache.insert(make_pair(seedVec, matListGpu_int8));

        delete matList_int8;
    }

    start = clock();
    iter_mineBytom(blockheader, 32, result);
    end = clock();
    std::cout << "mineBytom: "
              << (double)(end - start) / CLOCKS_PER_SEC << "s"
              << std::endl;

    if(seedCache.size() > cacheSize) {
        seedVec = seed_list.at(cacheSize - 1);
        seed_list.erase(seed_list.begin() + cacheSize - 1);
        seedCache.erase(seedVec);
    }

    mtx.unlock();
    return result;
}
