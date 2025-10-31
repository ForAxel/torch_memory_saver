#pragma once

// Define platform macros and include appropriate headers
#if defined(USE_ROCM)
// Include HIP runtime headers for AMD ROCm platform
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cassert>
/*
 * ROCm API Mapping References:
 * - CUDA Driver API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Driver_API_functions_supported_by_HIP.html
 * - CUDA Runtime API to HIP: https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/MUSA_Runtime_API_functions_supported_by_HIP.html
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
#define muMemGetAllocationGranularity hipMemGetAllocationGranularity
#define muMemUnmap hipMemUnmap
#define muMemRelease hipMemRelease
#define musaMallocHost hipHostMalloc
#define musaMemcpy hipMemcpy
// --- Memory Copy Direction Constants ---
#define musaMemcpyDeviceToHost hipMemcpyDeviceToHost
// --- Device and Stream Types ---
#define MUdevice hipDevice_t
#define musaStream_t hipStream_t
// --- Memory Allocation Constants ---
// Chunk size for memory creation operations (2 MB)
#define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
// --- Utility Macros ---
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ============================================================================
// CUDA Platform Configuration (NVIDIA GPUs)
// ============================================================================
#elif defined(USE_CUDA)
#include <musa_runtime_api.h>
#include <musa.h>

// ============================================================================
// Error: No Platform Specified
// ============================================================================
#else
#error "USE_PLATFORM is not set"
#endif
