/**
 * @file compare.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CE05A256_23CB_4243_8839_B1FDA9C540D2
#define CE05A256_23CB_4243_8839_B1FDA9C540D2

#include <stdint.h>
#include <stdlib.h>

#include "compare.cu_hip.hh"
#include "compare.dp.hh"
#include "compare.dpl.hh"
#include "compare.stl.hh"
#include "compare.thrust.hh"
#include "cusz/type.h"
#include "detail/busyheader.hh"

namespace psz {

template <pszpolicy P, typename T>
bool identical(T* d1, T* d2, size_t const len)
{
  if (P == SEQ)
    psz::cppstl_identical(d1, d2, len);
  else if (P == THRUST_DPL)
    thrustgpu_identical(d1, d2, len);
  else {
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
  }
}

template <pszpolicy P, typename T>
[[deprecated]] void probe_extrema(T* in, size_t len, T res[4])
{
  if (P == SEQ) psz::cppstl_extrema(in, len, res);
#ifdef REACTIVATE_THRUST_DPLGPU
  else if (P == THRUST_DPL)
    thrustgpu::thrustgpu_get_extrema_rawptr(in, len, res);
#endif
  else if (P == CUDA or P == ROCM) {
    psz::cu_hip::extrema(in, len, res);
  }
  else if (P == SYCL) {
    psz::dpcpp::extrema(in, len, res);
  }
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
}

template <typename T1, psz_runtime R = CUDA, typename T2 = T1>
void GPU_probe_extrema(T1* in, size_t len, T2& max_value, T2& min_value, T2& range)
{
  T1 result[4];

#ifndef PSZ_2505_MERGE
  if (R == CUDA or R == ROCM)  //
    psz::cu_hip::extrema(in, len, result);
#else
  if (R == CUDA or R == ROCM)
    module::GPU_extrema(in, len, result);
  else if (R == SYCL)
    dpcpp::GPU_extrema(in, len, result);
#endif
#ifdef REACTIVATE_THRUSTGPU
  else if (R == THRUST_DPL)
    thrustgpu::GPU_extrema(in, len, result);
#endif
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");

  min_value = result[0];
  max_value = result[1];
  range = max_value - min_value;
}

template <pszpolicy P, typename T>
bool error_bounded(
    T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr)
{
  bool eb_ed = true;
  if (P == SEQ) eb_ed = psz::cppstl_error_bounded(a, b, len, eb, first_faulty_idx);
#ifdef REACTIVATE_THRUST_DPLGPU
  else if (P == THRUST_DPL)
    eb_ed = psz::thrustgpu::thrustgpu_error_bounded(a, b, len, eb, first_faulty_idx);
#endif
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
  return eb_ed;
}

template <pszpolicy P, typename T>
void assess_quality(psz_statistics* s, T* xdata, T* odata, size_t const len)
{
  // [TODO] THRUST_DPL is not activated in the frontend
  if (P == SEQ)
    psz::cppstl_assess_quality(s, xdata, odata, len);
  else if (P == THRUST_DPL)
    psz::thrustgpu_assess_quality(s, xdata, odata, len);
  else if (P == SYCL) {
#if defined(PSZ_USE_1API)
    if constexpr (std::is_same_v<T, f4>) { psz::dpl_assess_quality(s, xdata, odata, len); }
    else {
      static_assert(std::is_same_v<T, f4>, "No f8, fast fail on sycl::aspects::fp64.");
    }
#endif
  }
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
}

}  // namespace psz

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
