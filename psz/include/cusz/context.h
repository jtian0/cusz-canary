
/**
 * @file context.h
 * @author Jiannan Tian
 * @brief Argument parser (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef A93D242E_0C7C_44BD_AE43_B3A26084971A
#define A93D242E_0C7C_44BD_AE43_B3A26084971A

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/type.h"
#include "stdint.h"

struct psz_cli_config {
  // filenames
  char opath[200];
  char file_input[500];
  char file_compare[500];

  // str for metadata
  char char_mode[4];
  char char_meta_eb[16];
  char char_predictor_name[sizeof("lorenzo-zigzag")];
  char char_hist_name[sizeof("histogram-centrality")];
  char char_codec1_name[sizeof("huffman-revisit")];
  char char_codec2_name[sizeof("huffman-revisit")];

  // dump intermediate
  bool dump_quantcode;
  bool dump_hist;
  bool dump_full_hf;

  bool task_construct;
  bool task_reconstruct;

  bool rel_range_scan;

  bool use_gpu_verify;

  bool skip_tofile;
  bool skip_hf;

  bool report_time;
  bool report_cr;
  bool verbose;
};
typedef psz_cli_config psz_cli_config;

struct psz_context {
  bool task_construct{false};
  bool task_reconstruct{false};
  bool task_dryrun{false};
  bool task_experiment{false};

  bool prep_binning{false};
  //   bool prep_logtransform{false};
  bool prep_prescan{false};

  bool use_demodata{false};
  bool use_autotune_hf{true};
  bool use_gpu_verify{false};

  bool skip_tofile{false};
  bool skip_hf{false};

  bool report_time{false};
  bool report_cr{false};
  bool report_cr_est{false};
  bool verbose{false};

  //   pszdevice device;

  // TODO: (need fix) if no default is specified, empty 0 -> lorenzo
  psz_predtype pred_type{Spline};
  char dbgstr_pred[10];

  // sizes
  uint32_t x{1}, y{1}, z{1}, w{1};
  size_t data_len{1};
  size_t splen{0};
  int ndim{-1};

  // filenames
  char demodata_name[40];
  char infile[500];
  char original_file[500];
  char opath[200];

  // pipeline config
  psz_dtype dtype{F4};
  psz_mode mode{Rel};
  double eb{0.0};
  double rel_eb{0.0};
  int dict_size{256}, radius{128};
  int quant_bytewidth{1}, huff_bytewidth{4};
  bool use_huffman{true};

  // spv gather-scatter config, tmp. unused
  float nz_density{0.2};
  float nz_density_factor{5};

  // codec config
  uint32_t codecs_in_use{0b01};
  int vle_sublen{512}, vle_pardeg{-1};

  // i/Hi
  INTERPOLATION_PARAMS intp_param;
};

typedef struct psz_context psz_context;
typedef psz_context pszctx;
typedef psz_context psz_manager;
typedef psz_context psz_resource;
typedef psz_context psz_arguments;

void capi_psz_version();
void capi_psz_versioninfo();

// Return a pszctx instance with default values.
pszctx* pszctx_default_values();

// Modify an empty pszctx with default values.
void pszctx_set_default_values(pszctx*);

// Use a minimal workset as the return object.
pszctx* pszctx_minimal_workset(
    psz_dtype const dtype, psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec);

void pszctx_set_rawlen(pszctx* ctx, size_t _x, size_t _y, size_t _z, size_t _w);
void pszctx_set_len(pszctx* ctx, pszlen len);
#define get_len3 pszctx_get_len3
void pszctx_create_from_argv(pszctx* ctx, int const argc, char** const argv);
void pszctx_create_from_string(pszctx* ctx, const char* in_str, bool dbg_print);

#ifdef PSZ_2505_MERGE

unsigned int CLI_x(psz_arguments* args);
unsigned int CLI_y(psz_arguments* args);
unsigned int CLI_z(psz_arguments* args);
unsigned int CLI_w(psz_arguments* args);
unsigned short CLI_radius(psz_arguments* args);
unsigned short CLI_bklen(psz_arguments* args);
psz_dtype CLI_dtype(psz_arguments* args);
psz_predtype CLI_predictor(psz_arguments* args);
psz_histotype CLI_hist(psz_arguments* args);
psz_codectype CLI_codec1(psz_arguments* args);
psz_codectype CLI_codec2(psz_arguments* args);
psz_mode CLI_mode(psz_arguments* args);
double CLI_eb(psz_arguments* args);

#endif

#ifdef __cplusplus
}
#endif

#endif /* A93D242E_0C7C_44BD_AE43_B3A26084971A */
