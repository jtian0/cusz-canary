/**
 * @file spline.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef AA9BE6AD_ECA4_4267_A97F_B12C25A2B0C1
#define AA9BE6AD_ECA4_4267_A97F_B12C25A2B0C1

#include <cstddef>
#include <cstdint>

// #include "mem/cxx_memobj.h"
#include "mem/cxx_memobj.h"

template <typename T>
using memobj = _portable::memobj<T>;

// template <typename T, typename E, typename FP = T>
// int spline_construct(
//     pszmem_cxx<T>* data, pszmem_cxx<T>* anchor, pszmem_cxx<E>* errctrl, void* _outlier, double
//     eb, double rel_eb, uint32_t radius, INTERPOLATION_PARAMS& intp_param, float* time, void*
//     stream, pszmem_cxx<T>* profiling_errors);

// template <typename T, typename E, typename FP = T>
// int spline_reconstruct(
//     pszmem_cxx<T>* anchor, pszmem_cxx<E>* errctrl, pszmem_cxx<T>* xdata, T* outlier_tmp, double
//     eb, uint32_t radius, INTERPOLATION_PARAMS intp_param, float* time, void* stream);

namespace psz {

template <typename T, typename E, typename Fp = T>
struct GPU_spline_construct {
  static int kernel_v0(
      memobj<T>* data, memobj<T>* anchor, memobj<E>* errctrl, void* _outlier, double eb,
      double rel_eb, uint32_t radius, INTERPOLATION_PARAMS& intp_param, float* time, void* stream,
      memobj<T>* profiling_errors);
};

template <typename T, typename E, typename Fp = T>
struct GPU_spline_reconstruct {
  static int kernel_v0(
      memobj<T>* anchor, memobj<E>* errctrl, memobj<T>* xdata, T* outlier_tmp, double eb,
      uint32_t radius, INTERPOLATION_PARAMS intp_param, float* time, void* stream);
};

}  // namespace psz

#endif /* AA9BE6AD_ECA4_4267_A97F_B12C25A2B0C1 */
