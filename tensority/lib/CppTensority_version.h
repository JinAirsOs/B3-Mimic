#pragma once
#ifndef __CPPTENSORITY_VERSION_H__
#define __CPPTENSORITY_VERSION_H__

#ifndef TENSORITY_EXPORT
#define TENSORITY_API __declspec(dllexport)
#else
#define TENSORITY_API __declspec(dllimport)
#endif // !TENSORITY_API

#include <string>
using namespace std;

typedef struct {
	string deviceName;
	int deviceVersion;
}driverVersionInfo;

#ifdef  __cplusplus
extern "C"
{
	//unsigned char TENSORITY_API *SimdTs2(unsigned char blockheader[32], unsigned char seed[32]);
	int TENSORITY_API *SimdTs2(unsigned char blockheader[32], unsigned char seed[32], unsigned char res[32]);

	TENSORITY_API int GetDeviceCount();
	TENSORITY_API driverVersionInfo *  GetDeviceDriverVersion(int, driverVersionInfo *);

}
#endif //  __cplusplus




#endif