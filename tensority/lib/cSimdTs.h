#ifndef _C_SIMD_TENSOR_H_
#define _C_SIMD_TENSOR_H_

#ifdef __cplusplus
extern "C" {
#endif
	#include <stdint.h>

	uint8_t *SimdTs(unsigned char blockheader[32], unsigned char seed[32]);

#ifdef __cplusplus
}
#endif

#endif
