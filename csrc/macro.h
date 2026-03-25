#pragma once

#if defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cassert>
/*
 * ROCm API Mapping References:
 * - CUDA Driver API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Driver_API_functions_supported_by_HIP.html
 * - CUDA Runtime API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html
 */
// --- Error Handling Types and Constants ---
#define MUresult hipError_t
#define musaError_t hipError_t
#define MUSA_SUCCESS hipSuccess
#define musaSuccess hipSuccess
// --- Error Reporting Functions ---
#define muGetErrorString hipDrvGetErrorString
#define musaGetErrorString hipGetErrorString
// --- Memory Management Functions ---
#define MUdeviceptr hipDeviceptr_t
#define muMemGetAllocationGranularity hipMemGetAllocationGranularity
#define muMemAddressReserve hipMemAddressReserve
#define muMemAddressFree hipMemAddressFree
#define muMemMap hipMemMap
#define muMemUnmap hipMemUnmap
#define muMemRelease hipMemRelease
#define musaMalloc hipMalloc
#define musaFree hipFree
#define musaMallocHost hipHostMalloc
#define musaFreeHost hipFreeHost
#define musaMemcpy hipMemcpy
#define musaMemGetInfo hipMemGetInfo
#define musaDeviceSynchronize hipDeviceSynchronize
// --- Memory Copy Direction Constants ---
#define musaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define musaMemcpyHostToDevice hipMemcpyHostToDevice
// --- Device and Stream Types ---
#define MUdevice hipDevice_t
#define musaStream_t hipStream_t
// --- Error codes ---
#define musaErrorMemoryAllocation hipErrorOutOfMemory
// --- Memory Allocation Handle ---
#define MUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
// --- Chunk size for memory creation operations (2 MB) ---
#define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
// --- Utility Macros ---
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// --- ROCm Version Feature Flags ---
// ROCm 6.x has hipMemCreate bug, requires chunked allocation workaround
// ROCm 7.0+ has fixed the bug, can use non-chunked allocation like CUDA
#if HIP_VERSION < 70000000
    #define TMS_ROCM_LEGACY_CHUNKED 1
#else
    #define TMS_ROCM_LEGACY_CHUNKED 0
#endif

#elif defined(USE_MUSA)
#include <musa_runtime_api.h>
#include <musa.h>

#define TMS_ROCM_LEGACY_CHUNKED 0

#else
#error "USE_PLATFORM is not set"
#endif
