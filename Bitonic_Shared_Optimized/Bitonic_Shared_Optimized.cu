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

static int* Data = 0;

__device__ void cudaSwap(int* a, int* b)
{
	int temp = *a;
	*a = *b;
	*b = temp;
}

__global__ void SortKernel(int* array, uint64_t size, uint64_t originalSize)
{
	uint64_t threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t step = size / 2;
	uint64_t startIdx = ((threadId / step) * size) + (threadId % step);
	uint64_t endIdx = startIdx + step;
	// false is down, true is up
	bool direction = ((threadId / (originalSize / 2)) % 2) == 0 ? false : true;

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

__global__ void SortKernelShared(int* array, uint64_t size, uint64_t originalSize, int numThreads)
{
	uint64_t GlobalThreadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t step = size / 2;

	// false is down, true is up
	bool direction = ((GlobalThreadId / (originalSize / 2)) % 2) == 0 ? false : true;
	//Range of numbers each thread will be responsible for adding 
	uint64_t range = (size/2)  / numThreads;
	__shared__ int sharedArr[2048]; // 32,768 Bytes. Cannot exceed this without going over 48k

	//Load memory into shared L1 cache from global
	int localIdx = threadIdx.x;
	uint64_t globalIdx = GlobalThreadId;
	int i = 0;
	do
	{
		// Get Indexes
		int startIdx = ((localIdx / step) * size) + (localIdx % step);
		uint64_t GlobalStartIdx = ((globalIdx / step) * size) + (globalIdx % step);
		//Put global data into shared memory
		sharedArr[startIdx] = array[GlobalStartIdx];
		sharedArr[startIdx + step] = array[GlobalStartIdx + step];
		//Increment by number of threads
		localIdx += numThreads;
		globalIdx += numThreads;
		//Increment i by range
		i++;
	} while (i < range);

	__syncthreads(); //Wait for all threads to finish loading from global memory

	//Begin operation 
	int stageSize = size;
	int stageStep = stageSize/2;
	int endIdx, startIdx;
	localIdx = threadIdx.x;
	int adjustedRange = range;
	while (stageSize != 1)
	{
		i = 0;
		do
		{
			startIdx = ((localIdx / stageStep) * stageSize) + (localIdx % stageStep);
			endIdx = startIdx + stageStep;
			if (endIdx >= 8192) break;
			if (direction) //If swapping upwards
			{
				if (sharedArr[startIdx] < sharedArr[endIdx])
				{
					cudaSwap(&sharedArr[startIdx], &sharedArr[endIdx]);
				}
			}
			else
			{
				if (sharedArr[startIdx] > sharedArr[endIdx])
				{
					cudaSwap(&sharedArr[startIdx], &sharedArr[endIdx]);
				}

			}
			localIdx += numThreads;
			i++;
		} while (i < adjustedRange);

		//Wait for all threads to finished before continuing to the next level
		__syncthreads();
		stageSize = stageStep;
		stageStep = stageSize / 2;
		//adjustedRange *= 2;
		localIdx = threadIdx.x;
	}

	localIdx = threadIdx.x;
	globalIdx = GlobalThreadId;
	i = 0;
	do
	{
		// Get Indexes
		startIdx = ((localIdx / step) * size) + (localIdx % step);
		uint64_t GlobalStartIdx = ((globalIdx / step) * size) + (globalIdx % step);
		//Put global data into shared memory
		array[GlobalStartIdx] = sharedArr[startIdx];
		array[GlobalStartIdx + step] = sharedArr[startIdx + step];
		//Increment by number of threads
		localIdx += numThreads;
		globalIdx += numThreads;
		//Increment i by range
		i++;
	} while (i < range);

	__syncthreads();
}

void SharedBitonic(int* aData, uint64_t aSize)
{
	cudaError_t cudaStatus;
	int* dev_array = 0;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return ;
	}

	//Allocate memory on device (GPU)
	cudaStatus = cudaMalloc((void**)&dev_array, aSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_array);
		return ;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_array, aData, aSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_array);
		return ;
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
	printf("Shared Memory per Multiprocessor %d\n", prop.sharedMemPerMultiprocessor);
	int maxThreads = prop.maxThreadsPerBlock;
	int maxBlocks = ((aSize / 2) + maxThreads - 1) / maxThreads;
	if (maxThreads > aSize / 2)
	{
		maxBlocks = 1;
		maxThreads = aSize / 2;
	}
	printf("  Max Threads per block: %d\n  Max Blocks: %d\n", maxThreads, maxBlocks);

	dim3 threads(maxThreads, 1);
	dim3 blocks(maxBlocks, 1);
	cudaFuncSetCacheConfig(SortKernelShared, cudaFuncCachePreferL1);
	{
		Timer T;
		for (uint64_t i = 2; i <= aSize; i *= 2)
		{
			//printf("next iter\n");
			uint64_t j = i;
			while (j != 1)
			{
				// 48Kb of shared memory / 12,000 int32 values stored. 
				// 8,192 entries shared to without going over 48k in shared memory
				// 4,096 
				if (j <= std::pow(2, 11)) 
				{

					SortKernelShared << <blocks, threads >> > (dev_array, j, i, maxThreads);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "SortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
						cudaFree(dev_array);
						return;
					}
					break;
				}
				else
				{
					//printf("size: %d \n", j);
					SortKernel << <blocks, threads >> > (dev_array, j, i);
					// Check for any errors launching the kernel
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "SortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
						cudaFree(dev_array);
						return;
					}
				}
				j = j / 2;
			}
			// REMEMBER TO REMOVE
			cudaStatus = cudaMemcpy(aData, dev_array, aSize * sizeof(int), cudaMemcpyDeviceToHost);
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
		return;
	}
	printf("Sync Completed\n");
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(aData, dev_array, aSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_array);
		return;
	}
}


int main()
{
	printf("Running Shared Optimized\n");


	// Declare range of random numbers
	const int range = 10;

	//Declare problem size
	const uint64_t size = pow(2, 16);
	printf("Sorting %llu values\n", size);
	// Allocate memory on host device
	Data = (int*)std::malloc(size * sizeof(int));
	printf("Allocating %llu bytes of memory\n", size * sizeof(int));
	// randomly fill array
	for (uint64_t i = 0; i < size; i++)
	{
		Data[i] = rand() % range;
	}

	{
		Timer T;
		SharedBitonic(Data, size);
		printf("Execution time: ");
	}

	printf("Transfer Completed\n");
	printf("confirming solition\n");
	int prev = 0;
	bool passed = true;
	for (uint64_t i = 0; i < size; i++)
	{
		if (Data[i] < prev)
		{
			//printf("failed at %llu Current %llu, Prev %llu\n", i, Data[i], prev);
			passed = false;
		}
		prev = Data[i];
	}

	if (passed) printf("All %llu number correctly sorted\n", size);
	else printf("Failed, incorrect sorting value\n");
	
	printf("Printing results\n");
	std::ofstream out1("Output1.txt");
	prev = 0;
	for (int i = 0; i < size; i++)
	{
		out1 << Data[i] << " ";
		if (Data[i] < prev ) out1 << "\n\n\n";
		prev = Data[i];
	}
	
}