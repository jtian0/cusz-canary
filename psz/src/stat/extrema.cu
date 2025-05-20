#include <cuda_runtime.h>

#include "cusz/type.h"
#include "detail/compare.cu_hip.hh"
#include "detail/port.hh"
#include "utils/err.hh"
// definitions
#include "detail/extrema.cuhip.inl"

template void psz::cu_hip::extrema<f4>(f4* d_ptr, szt len, f4 res[4]);
template void psz::cu_hip::extrema<f8>(f8* d_ptr, szt len, f8 res[4]);