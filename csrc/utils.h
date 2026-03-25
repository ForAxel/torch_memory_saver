#pragma once
#include <iostream>
#include <vector>
#include "macro.h"

// #define TMS_DEBUG_LOG

// Cannot use pytorch (libc10.so) since LD_PRELOAD happens earlier than `import torch`
// Thus copy from torch Macros.h
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

#define SIMPLE_CHECK(COND, MSG) \
  do { \
    if (!(COND)) { \
        std::cerr << "[torch_memory_saver.cpp] " << MSG \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define MURESULT_CHECK(EXPR) \
  do { \
    MUresult __result = (EXPR); \
    if (__result != MUSA_SUCCESS) { \
        const char* err_str = nullptr; \
        muGetErrorString(__result, &err_str); \
        std::cerr << "[torch_memory_saver.cpp] MUresult error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define MUSA_ERROR_CHECK(EXPR) \
  do { \
    musaError_t __result = (EXPR); \
    if (__result != musaSuccess) { \
        const char* err_str = musaGetErrorString(__result); \
        std::cerr << "[torch_memory_saver.cpp] musaError error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)


namespace MUSAUtils {
#if defined(USE_ROCM)

    #if HIP_VERSION < 60304000 // rocm/hip 6.3.4
        #pragma message "You need to implement torch_memory_saver in ROCm/HIP 6.3.4 or lower. We did not support it currently."
    #else
        #if TMS_ROCM_LEGACY_CHUNKED
            #pragma message "Using ROCm/HIP 6.x implementation (chunked allocation workaround)"
        #else
            #pragma message "Using ROCm/HIP >= 7.0 implementation (single allocation, same as MUSA)"
        #endif

        static MUdevice mu_ctx_get_device() {
            int device;
            MUSA_ERROR_CHECK(hipGetDevice(&device));
            return static_cast<MUdevice>(device);
        }

        static MUdevice mu_device_get(int device_ordinal) {
            MUdevice ans;
            MURESULT_CHECK(hipDeviceGet(&ans, device_ordinal));
            return ans;
        }

        static musaError_t mu_mem_create(MUmemGenericAllocationHandle *alloc_handle, size_t size, MUdevice device) {
            hipMemAllocationProp prop = {};
            prop.type = hipMemAllocationTypePinned;
            prop.location.type = hipMemLocationTypeDevice;
            prop.location.id = device;
            prop.allocFlags.compressionType = 0x0;

            hipError_t ret = hipMemCreate(alloc_handle, size, &prop, 0);
            if (ret == hipErrorOutOfMemory) {
                std::cerr << "[torch_memory_saver.cpp] hipMemCreate hipErrorOutOfMemory (may not be an issue e.g. torch allocator will free cache and retry)" << std::endl;
                return hipErrorOutOfMemory;
            }
            MURESULT_CHECK(ret);

            return hipSuccess;
        }

        static void mu_mem_set_access(void *ptr, size_t size, MUdevice device) {
            hipMemAccessDesc accessDesc = {};
            accessDesc.location.type = hipMemLocationTypeDevice;
            accessDesc.location.id = device;
            accessDesc.flags = hipMemAccessFlagsProtReadWrite;
            MURESULT_CHECK(hipMemSetAccess(ptr, size, &accessDesc, 1));
        }
    #endif

#elif defined(USE_MUSA)
    static musaError_t mu_mem_create(MUmemGenericAllocationHandle *alloc_handle, size_t size, MUdevice device) {
        MUmemAllocationProp prop = {};
        prop.type = MU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;

        int flag = 0;
        MURESULT_CHECK(muDeviceGetAttribute(&flag, MU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_MUSA_VMM_SUPPORTED, device));
        if (flag) {  // support GPUDirect RDMA if possible
            prop.allocFlags.gpuDirectRDMACapable = 1;
        }

        MUresult ret = muMemCreate(alloc_handle, size, &prop, 0);
        if (ret == MUSA_ERROR_OUT_OF_MEMORY) {
            std::cerr << "[torch_memory_saver.cpp] muMemCreate MUSA_ERROR_OUT_OF_MEMORY (may not be an issue e.g. torch allocator will free cache and retry)" << std::endl;
            return musaErrorMemoryAllocation;
        }
        MURESULT_CHECK(ret);

        return musaSuccess;
    }

    static void mu_mem_set_access(void *ptr, size_t size, MUdevice device) {
        MUmemAccessDesc access_desc = {};
        access_desc.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device;
        access_desc.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        MURESULT_CHECK(muMemSetAccess((MUdeviceptr) ptr, size, &access_desc, 1));
    }

    static MUdevice mu_ctx_get_device() {
        MUdevice ans;
        MURESULT_CHECK(muCtxGetDevice(&ans));
        return ans;
    }

    static MUdevice mu_device_get(int device_ordinal) {
        MUdevice ans;
        MURESULT_CHECK(muDeviceGet(&ans, device_ordinal));
        return ans;
    }

#else
    #error "USE_PLATFORM is not set"

#endif
}

inline bool get_bool_env_var(const char* name) {
    const char* env_cstr = std::getenv(name);
    if (env_cstr == nullptr) {
        return false;
    }

    std::string env_str(env_cstr);
    if (env_str == "1" || env_str == "true" || env_str == "TRUE" || env_str == "yes" || env_str == "YES") {
        return true;
    }
    if (env_str == "0" || env_str == "false" || env_str == "FALSE" || env_str == "no" || env_str == "NO") {
        return false;
    }

    std::cerr << "[torch_memory_saver.cpp] Unsupported environment varialbe value "
              << " name=" << name << " value=" << env_str
              << std::endl;
    exit(1);
}
