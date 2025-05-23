/**
 * @file type.h
 * @author Jiannan Tian
 * @brief C-complient type definitions; no methods in this header.
 * @version 0.3
 * @date 2022-04-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef PSZ_TYPE_H
#define PSZ_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

typedef _portable_device psz_device;
typedef _portable_runtime psz_runtime;
typedef _portable_runtime psz_backend;
typedef _portable_toolkit psz_toolkit;

typedef psz_runtime psz_policy_deprecated;
typedef psz_runtime psz_execution_policy;
typedef psz_policy_deprecated pszpolicy;

typedef _portable_stream_t psz_stream_t;
typedef _portable_mem_control psz_mem_control;
typedef _portable_dtype psz_dtype;
typedef _portable_len3 psz_len3;
typedef _portable_size3 psz_size3;
typedef _portable_data_summary psz_data_summary;

typedef psz_execution_policy psz_platform;

typedef void* uninit_stream_t;

//////// state enumeration

typedef enum psz_error_status {  //
  PSZ_SUCCESS,
  PSZ_WARN_RADIUS_TOO_LARGE,
  PSZ_WARN_OUTLIER_TOO_MANY,
  PSZ_WARN_INCOMPRESSIABLE,
  PSZ_ABORT_UNSUPPORTED_TYPE,
  PSZ_ABORT_UNSUPPORTED_DIMENSION,
  PSZ_ABORT_NOT_IMPLEMENTED,
} psz_error_status;
typedef psz_error_status pszerror;

// 2505 MERGE
#define CUSZ_SUCCESS PSZ_SUCCESS

// aliasing
typedef uint8_t byte_t;
typedef size_t szt;

#define DEFAULT_PREDICTOR Lorenzo
#define DEFAULT_HISTOGRAM HistogramGeneric
#define DEFAULT_CODEC Huffman
#define NULL_HISTOGRAM NullHistogram
#define NULL_CODEC NullCodec

#ifndef PSZ_2505_MERGE
typedef enum psz_space  //
{ Device = 0,
  Host = 1,
  None = 2 } psz_space;
#endif

typedef enum psz_mode { Abs, Rel, Verbatim } psz_mode;
typedef enum { Lorenzo, LorenzoZigZag, LorenzoProto, Spline } psz_predtype;

typedef enum {
  FP64toFP32,
  LogTransform,
  ShiftedLogTransform,
  Binning2x2,
  Binning2x1,
  Binning1x2,
} psz_prep_type;

typedef enum {
  Huffman,
  HuffmanRevisit,
  FZGPUCodec,
  RunLength,
  NullCodec,
} psz_codectype;

#ifndef PSZ_2505_MERGE
typedef enum psz_hfbktype  //
{ Canonical = 1,
  Sword = 2,
  Mword = 3 } psz_hfbktype;

typedef enum psz_hfpartype  //
{ Coarse = 0,
  Fine = 1 } psz_hfpartype;
#endif

typedef enum {
  HistogramGeneric,
  HistogramSparse,
  NullHistogram,
} psz_histotype;

#ifndef PSZ_2505_MERGE
//////// configuration template
typedef struct pszlen {
  // clang-format off
    union { size_t x0, x; };
    union { size_t x1, y; };
    union { size_t x2, z; };
    union { size_t x3, w; };
  // clang-format on
} pszlen;
#endif

#ifndef PSZ_2505_MERGE
typedef struct pszpredictor {
  psz_predtype type;
} pszpredictor;
#endif

typedef struct psz_quantizer {
  int radius;
} psz_quantizer;
typedef psz_quantizer pszquantizer;

typedef struct psz_hfruntimeconfig {
  // psz_hfbktype book;
  // psz_hfpartype style;
  int booklen;
  int coarse_pardeg;
} psz_hfruntimeconfig;
typedef psz_hfruntimeconfig pszhfrc;

////// wrap-up

typedef struct psz_framework {
  pszpredictor predictor;
  pszquantizer quantizer;
  pszhfrc hfcoder;
  f4 max_outlier_percent;
} psz_framework;
typedef psz_framework pszframe;

struct psz_context;
typedef struct psz_context pszctx;

struct psz_header;
typedef struct psz_header pszheader;

typedef struct psz_compressor {
  void* compressor;
  pszctx* ctx;
  pszheader* header;
  pszframe* framework;
  psz_dtype type;
} psz_compressor;
typedef psz_compressor pszcompressor;

typedef struct psz_runtimeconfig {
  f8 eb;
  psz_mode mode;
  psz_predtype pred_type;
  bool est_cr;
} psz_runtimeconfig;
typedef psz_runtimeconfig pszrc;

typedef struct Res {
  f8 min, max, rng, std;
} pszscanres;
typedef pszscanres Res;

// typedef struct psz_summary {
//   // clang-format off
//     pszscanres odata, xdata;
//     struct { f8 PSNR, MSE, NRMSE, coeff; } score;
//     struct { f8 abs, rel, pwrrel; size_t idx; } max_err;
//     struct { f8 lag_one, lag_two; } autocor;
//     f8 user_eb;
//     size_t len;
//   // clang-format on
// } psz_summary;
// typedef psz_summary pszsummary;

// nested struct object (rather than ptr) results in Swig creating a `__get`,
// which can be breaking. Used `prefix_` instead.
typedef struct psz_statistics {
  psz_data_summary odata, xdata;
  f8 score_PSNR, score_MSE, score_NRMSE, score_coeff;
  f8 max_err_abs, max_err_rel, max_err_pwrrel;
  size_t max_err_idx;
  f8 autocor_lag_one, autocor_lag_two;
  f8 user_eb;
  size_t len;
} psz_statistics;

typedef u1* pszout;
// used for bridging some compressor internal buffer
typedef pszout* ptr_pszout;

// struct INTERPOLATION_PARAMS {
//   double alpha{1.75};
//   double beta{4.0};
//   // bool interpolators[3];
//   bool use_md[6];
//   bool use_natural[6];
//   bool reverse[6];
//   uint8_t auto_tuning{3};
//   INTERPOLATION_PARAMS() :
//       use_md{true, true, false, false, false, false},
//       use_natural{false, false, false, false, false, false},
//       reverse{false, false, false, false, false, false} {};
// };

typedef struct psz_interp_params {
  double alpha, beta;

  bool use_md[6];
  bool use_natural[6];
  bool reverse[6];
  uint8_t auto_tuning;
} psz_interp_params;

typedef struct psz_interp_params INTERPOLATION_PARAMS;

// C-style "constructor"
static inline psz_interp_params make_default_params(void)
{
  psz_interp_params p = {
      .alpha = 1.75,
      .beta = 4.0,
      .use_md = {1, 1, 0, 0, 0, 0},
      .use_natural = {0, 0, 0, 0, 0, 0},
      .reverse = {0, 0, 0, 0, 0, 0},
      .auto_tuning = 3};
  return p;
}

#ifdef __cplusplus
}
#endif

#endif
