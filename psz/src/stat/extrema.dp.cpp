#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/type.h"
#include "detail/port.hh"
#include "detail/stat/compare/compare.dp.hh"
#include "utils/err.hh"
// definitions
#include "detail/extrema.dp.inl"

template void psz::dpcpp::extrema<f4>(f4* d_ptr, szt len, f4 res[4]);
// template void psz::dpcpp::extrema<f8>(f8* d_ptr, szt len, f8 res[4]);