#include <iostream>
#include "api_forwarder.h"
#include "utils.h"
#include "macro.h"

namespace APIForwarder {
    using MusaMallocFunc = musaError_t (*)(void**, size_t);
    using MusaFreeFunc = musaError_t (*)(void*);

#if defined(USE_ROCM)
    static constexpr const char* MALLOC_NAME = "hipMalloc";
    static constexpr const char* FREE_NAME = "hipFree";
#else
    static constexpr const char* MALLOC_NAME = "musaMalloc";
    static constexpr const char* FREE_NAME = "musaFree";
#endif

    static void *check_dlsym(void *value) {
        if (nullptr == value) {
            std::cerr << "[torch_memory_saver.cpp] dlsym failed dlerror=" << dlerror() << std::endl;
            exit(1);
        }
        return value;
    }

    static MusaMallocFunc real_musa_malloc_ = NULL;
    static MusaFreeFunc real_musa_free_ = NULL;

    musaError_t call_real_musa_malloc(void **ptr, size_t size) {
        if (C10_UNLIKELY(nullptr == real_musa_malloc_)) {
            real_musa_malloc_ = (MusaMallocFunc) check_dlsym(dlsym(RTLD_NEXT, MALLOC_NAME));
        }

        musaError_t ret = real_musa_malloc_(ptr, size);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] APIForwarder.call_real_musa_malloc "
                  << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }

    musaError_t call_real_musa_free(void *ptr) {
        if (C10_UNLIKELY(nullptr == real_musa_free_)) {
            real_musa_free_ = (MusaFreeFunc) check_dlsym(dlsym(RTLD_NEXT, FREE_NAME));
        }

        musaError_t ret = real_musa_free_(ptr);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] APIForwarder.call_real_musa_free "
                  << " ptr=" << ptr << " ret=" << ret
                  << std::endl;
#endif

        return ret;
    }
}
