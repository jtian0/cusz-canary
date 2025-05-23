#include "detail/compare.thrust.hh"
#include "detail/compare.thrust.inl"

#define THRUSTGPU_ASSESS(Tliteral, T)             \
  template void psz::thrustgpu_assess_quality<T>( \
      psz_statistics * s, T * xdata, T * odata, size_t const len);

THRUSTGPU_ASSESS(fp32, float);

#undef THRUSTGPU_ASSESS
