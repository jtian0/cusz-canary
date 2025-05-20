#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "detail/extrema_thrust.inl"
#include "detail/stat/compare/compare.dpl.hh"

#define THRUSTGPU_DESCRIPTION(Tliteral, T)         \
  template void psz::thrustgpu_get_extrema_rawptr( \
      T* d_ptr, size_t len, T res[4]);

THRUSTGPU_DESCRIPTION(fp32, float)

#undef THRUSTGPU_DESCRIPTION