CUDA_ARCH_FLAGS ?= -arch=sm_60 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 
CC_FLAGS += --std=c++11 $(CUDA_ARCH_FLAGS)

all: cSimdTs.a test_bytom 

cSimdTs.a: BytomPoW.o sha3.o byte_order.o cSimdTs.o
	ar crv $@ byte_order.o sha3.o cSimdTs.o BytomPoW.o
	mv cSimdTs.a ..
	cd .. && go test

test_bytom: dgemm.o test_bytom.o sha3.o byte_order.o BytomPoW.o 
	g++ -o test_bytom byte_order.c sha3.c  test_BytomPoW.cpp BytomPoW.cpp  dgemm.o -I. -lpthread -O3  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcudart

test_bytom.o: test_BytomPoW.cpp
	g++ -o $@ -c $< -I/usr/local/cuda/include

test_matrix : test_matrix.cpp matrix.o
	g++ -o $@ $< matrix.o -I/usr/local/cuda/include  -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcudart -lpthread -O3 

sha3.o : sha3.c
	g++ -o $@ -c $< -fPIC


byte_order.o : byte_order.c
	g++ -o $@ -c $< 

BytomPoW.o : BytomPoW.cu
	nvcc  -o $@ -c $< $(CC_FLAGS) -I/usr/local/cuda/include -lpthread -O3 --compiler-options '-fPIC'

matrix.o : matrix.cu
	nvcc  -o $@ -c $< $(CC_FLAGS) -I/usr/local/cuda/include -lpthread -O3 --compiler-options '-fPIC'

cSimdTs.o: cSimdTs.cpp
	g++ -std=c++11 -o $@ -c $< -fPIC -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcudart -lpthread -O3 

clean:
	rm *.o test_bytom cSimdTs.so


