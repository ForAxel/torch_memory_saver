#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

#if defined(USE_ROCM)
#include "hardware_amd_support.h"
#endif

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

musaError_t TorchMemorySaver::malloc(void **ptr, MUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
#if defined(USE_ROCM)
    return ROCmHIPImplementation::rocm_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    MUmemGenericAllocationHandle allocHandle;

    musaError_t ret = MUSAUtils::mu_mem_create(&allocHandle, size, device);
    if (ret != musaSuccess) {
        return ret;
    }

    MURESULT_CHECK(muMemAddressReserve((MUdeviceptr *) ptr, size, 0, 0, 0));
    MURESULT_CHECK(muMemMap((MUdeviceptr) * ptr, size, 0, allocHandle, 0));
    MUSAUtils::mu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.musa_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return musaSuccess;
}

musaError_t TorchMemorySaver::free(void *ptr) {
#if defined(USE_ROCM)
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_musa_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    MURESULT_CHECK(muMemUnmap((MUdeviceptr) ptr, metadata.size));
    MURESULT_CHECK(muMemRelease(metadata.allocHandle));
    MURESULT_CHECK(muMemAddressFree((MUdeviceptr) ptr, metadata.size));

    if (nullptr != metadata.cpu_backup) {
        MUSA_ERROR_CHECK(musaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.musa_free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return musaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag) {
#if defined(USE_ROCM)
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                MUSA_ERROR_CHECK(musaMallocHost(&metadata.cpu_backup, metadata.size));
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use musaMemcpyAsync if needed
            MUSA_ERROR_CHECK(musaMemcpy(metadata.cpu_backup, ptr, metadata.size, musaMemcpyDeviceToHost));
        }

        MURESULT_CHECK(muMemUnmap((MUdeviceptr) ptr, metadata.size));
        MURESULT_CHECK(muMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if defined(USE_ROCM)
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        MUmemGenericAllocationHandle newAllocHandle;
        MUSA_ERROR_CHECK(MUSAUtils::mu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        MURESULT_CHECK(muMemMap((MUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

        MUSAUtils::mu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use musaMemcpyAsync if needed
            MUSA_ERROR_CHECK(musaMemcpy(ptr, metadata.cpu_backup, metadata.size, musaMemcpyHostToDevice));
            // maybe we can free host memory if needed (currently keep it there to reduce re-alloc time)
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                  << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}