#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>

class Timer {
public:
	std::chrono::system_clock::time_point Begin;
	std::chrono::system_clock::time_point End;
	std::chrono::system_clock::duration RunTime;
	Timer() {//constructor
		Begin = std::chrono::system_clock::now();
	}
	~Timer() {
		End = std::chrono::system_clock::now();
		RunTime = End - Begin;
		printf("%llu us\n", std::chrono::duration_cast<std::chrono::microseconds>(RunTime).count());
	}
};

__device__ void cudaSwap(int *a, int *b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

__global__ void SortKernel(int *array, uint64_t size, uint64_t originalSize)
{
	uint64_t threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t step = size / 2;
	uint64_t startIdx = ((threadId / step) * size) + (threadId % step);
	uint64_t endIdx = startIdx + step;
	// false is down, true is up
	bool direction = ((threadId / (originalSize/2)) % 2) == 0 ? false : true;

	if (direction) //If swapping upwards
	{
		if (array[startIdx] < array[endIdx])
		{
			cudaSwap(&array[startIdx], &array[endIdx]);
		}
	}
	else
	{
		if (array[startIdx] > array[endIdx])
		{
			cudaSwap(&array[startIdx], &array[endIdx]);
		}
	}
}


int main()
{

	cudaError_t cudaStatus;
	static int* Data = 0;
	int* dev_array = 0;

	// Declare range of random numbers
	const int range = 10;

	//Declare problem size
	const uint64_t size = pow(2, 30);

	printf("Data Prep Took ");
	{
		Timer T;
		// Allocate memory on host device
		//Data = (int*)std::malloc(size * sizeof(int));
		cudaMallocHost((int**)&Data, size * sizeof(int));
		printf("Allocating %llu bytes of memory\n", size * sizeof(int));
		// randomly fill array
		for (uint64_t i = 0; i < size; i++)
		{
			Data[i] = rand() % range;
		}

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			return cudaStatus;
		}

		//Allocate memory on device (GPU)
		cudaStatus = cudaMalloc((void**)&dev_array, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			cudaFree(dev_array);
			return cudaStatus;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_array, Data, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			cudaFree(dev_array);
			return cudaStatus;
		}

		// Get Device info
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		std::printf("Device Number: %d\n", 0);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		int maxThreads = prop.maxThreadsPerBlock;
		int maxBlocks = ((size / 2) + maxThreads - 1) / maxThreads;
		if (maxThreads > size / 2)
		{
			maxBlocks = 1;
			maxThreads = size / 2;
		}
		printf("  Max Threads per block: %d\n  Max Blocks: %d\n", maxThreads, maxBlocks);

		dim3 threads(maxThreads, 1);
		dim3 blocks(maxBlocks, 1);

		{
			Timer T;
			for (uint64_t stageSize = 2; stageSize <= size; stageSize *= 2)
			{
				//printf("next iter\n");
				uint64_t stepSize = stageSize;
				while (stepSize != 1)
				{
					//printf("size: %d \n", j);
					SortKernel << <blocks, threads >> > (dev_array, stepSize, stageSize);
					// Check for any errors launching the kernel
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "SortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
						cudaFree(dev_array);
						return cudaStatus;
					}
					stepSize /= 2;
				}
			}
			printf("Runtime took ");
		}
		printf("Processing Completed\n");
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SortKernel!\n", cudaStatus);
			cudaFree(dev_array);
			return cudaStatus;
		}
		printf("Sync Completed\n");
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(Data, dev_array, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			cudaFree(dev_array);
			return cudaStatus;
		}
	
		printf("Execution took ");
	}	

	printf("Transfer Completed\n");
	printf("confirming solition\n");
	int prev = 0;
	bool passed = true;
	for (uint64_t i = 0; i < size; i++)
	{
		if (Data[i] < prev)
		{
			printf("failed at %llu Current %llu, Prev %llu\n", i,Data[i], prev);
			passed = false;
		}
		prev = Data[i];
	}

	if (passed) printf("All %llu number correctly sorted\n", size);
	else printf("Failed, incorrect sorting value\n");
	/*
	printf("Printing results\n");
	std::ofstream out1("Output1.txt");;
	for (int i = 0; i < size; i++)
	{
		out1 << Data[i] << " ";
		if (i % 64 == 0) out1 << std::endl;
	}
	*/
}