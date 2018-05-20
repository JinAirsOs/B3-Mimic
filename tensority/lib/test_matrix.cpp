extern"C" {
    #include<cblas.h>
} 
#include "matrix.h"
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <algorithm>


int main(void) {
   clock_t start, end;
   test_gpu1();
   test_gpu1();
   test_gpu1();
   start = clock();
   test_gpu1();
   end = clock();
   cout << (double)(end - start) / 1000000 << "\n";
   end = clock();
   test_gpu2();
   start = clock();
   cout << (double)(start - end) / 1000000 << "\n";
}
