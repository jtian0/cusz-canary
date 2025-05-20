#include "detail/compare.thrust.hh"
#include "detail/extrema.thrust.inl"

#define THRUSTGPU_DESCRIPTION(Tliteral, T)                    \
  template void psz::thrustgpu::thrustgpu_get_extrema_rawptr( \
      T* d_ptr, size_t len, T res[4]);

THRUSTGPU_DESCRIPTION(fp32, float)

#undef THRUSTGPU_DESCRIPTION