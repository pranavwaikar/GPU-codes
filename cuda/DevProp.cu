// headers
#include <stdio.h>

int main(void)
{
    // function declarations
    void PrintCUDADeviceProperties(void);
    
    // code
    PrintCUDADeviceProperties();
}

void PrintCUDADeviceProperties(void)
{
	// function declarations
	int ConvertSMVersionNumberToCores(int, int);

	// code
	printf("CUDA INFORMATION :\n");
	printf("===========================================================================\n");

	cudaError_t ret_cuda_rt;
	int dev_count;
	ret_cuda_rt = cudaGetDeviceCount(&dev_count);
	if (ret_cuda_rt != cudaSuccess)
	{
		printf("CUDA Runtime API Error - cudaGetDeviceCount() Failed Due To %s. Exitting Now ...\n", cudaGetErrorString(ret_cuda_rt));
	}
	else if (dev_count == 0)
	{
		printf("There Is No CUDA Supprted Device On This System. Exitting Now ...\n");
		return;
	}
	else
	{
		printf("Total Number Of CUDA Supporting GPU Device/Devices On This System : %d\n", dev_count);
		for (int i = 0; i<dev_count; i++)
		{
			cudaDeviceProp dev_prop;
			int driverVersion = 0, runtimeVersion = 0;

			ret_cuda_rt = cudaGetDeviceProperties(&dev_prop, i);
			if (ret_cuda_rt != cudaSuccess)
			{
				printf("%s in %s at line %d\n", cudaGetErrorString(ret_cuda_rt), __FILE__, __LINE__);
				return;
			}
			printf("\n");
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);
			printf("******** CUDA DRIVER AND RUNTIME INFORMATION ********\n");
			printf("=====================================================\n");
			printf("CUDA Driver Version                                  : %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
			printf("CUDA Runtime Version                                 : %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
			printf("=====================================================\n");
			printf("********** GPU DEVICE GENERAL INFORMATION ***********\n");
			printf("=====================================================\n");
			printf("GPU Device Number                                    : %d\n", i);
			printf("GPU Device Name                                      : %s\n", dev_prop.name);
			printf("GPU Device Compute Capability                        : %d.%d\n", dev_prop.major, dev_prop.minor);
			printf("GPU Device Clock Rate                                : %d\n", dev_prop.clockRate);
			printf("GPU Device Type                                      : ");
			if (dev_prop.integrated)
				printf("Integrated ( On-Board )\n");
			else
				printf("Discrete ( Card )\n");
			printf("\n");
			printf("********** GPU DEVICE MEMORY INFORMATION ************\n");
			printf("=====================================================\n");
			printf("GPU Device Total Memory                              : %.0f GB = %.0f MB = %llu Bytes\n", ((float)dev_prop.totalGlobalMem / 1048576.0f) / 1024.0f, (float)dev_prop.totalGlobalMem / 1048576.0f, (unsigned long long) dev_prop.totalGlobalMem);
			printf("GPU Device Available Memory                          : %lu Bytes\n", (unsigned long)dev_prop.totalConstMem);
			printf("GPU Device Host Memory Mapping Capability            : ");
			if (dev_prop.canMapHostMemory)
				printf("Yes ( Can Map Host Memory To Device Memory )\n");
			else
				printf("No ( Can Not Map Host Memory To Device Memory )\n");
			printf("\n");
			printf("****** GPU DEVICE MULTIPROCESSOR INFORMATION ********\n");
			printf("=====================================================\n");
			printf("GPU Device Number Of SMProcessors                    : %d\n", dev_prop.multiProcessorCount);
			printf("GPU Device Number Of Cores Per SMProcessors          : %d\n", ConvertSMVersionNumberToCores(dev_prop.major, dev_prop.minor));
			printf("GPU Device Total Number Of Cores                     : %d\n", ConvertSMVersionNumberToCores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
			printf("GPU Device Shared Memory Per SMProcessor             : %lu\n", (unsigned long)dev_prop.sharedMemPerBlock);
			printf("GPU Device Number Of Registers Per SMProcessor       : %d\n", dev_prop.regsPerBlock);
			printf("\n");
			printf("*********** GPU DEVICE THREAD INFORMATION ***********\n");
			printf("=====================================================\n");
			printf("GPU Device Maximum Number Of Threads Per SMProcessor : %d\n", dev_prop.maxThreadsPerMultiProcessor);
			printf("GPU Device Maximum Number Of Threads Per Block       : %d\n", dev_prop.maxThreadsPerBlock);
			printf("GPU Device Threads In Warp                           : %d\n", dev_prop.warpSize);
			printf("GPU Device Maximum Thread Dimensions                 : ( %d, %d, %d )\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
			printf("GPU Device Maximum Grid Dimensions                   : ( %d, %d, %d )\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
			printf("\n");
			printf("*********** GPU DEVICE DRIVER INFORMATION ***********\n");
			printf("=====================================================\n");
			printf("GPU Device has ECC support                           : %s\n", dev_prop.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			printf("GPU Device CUDA Driver Mode ( TCC Or WDDM )          : %s\n", dev_prop.tccDriver ? "TCC ( Tesla Compute Cluster Driver )" : "WDDM ( Windows Display Driver Model )");
#endif
			printf("***************************************************************************\n");
		}
	}
}

int ConvertSMVersionNumberToCores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        { -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return(nGpuArchCoresPerSM[index - 1].Cores);
}
