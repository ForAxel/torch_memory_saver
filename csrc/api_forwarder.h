#pragma once
#include <dlfcn.h>
#include "macro.h"

namespace APIForwarder {
    musaError_t call_real_musa_malloc(void **ptr, size_t size);
    musaError_t call_real_musa_free(void *ptr);
}
