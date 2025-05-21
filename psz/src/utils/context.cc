/**
 * @file context.cc
 * @author Jiannan Tian
 * @brief context struct with argument parser
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#include "cusz/context.h"

#include <cstring>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "cusz/type.h"
#include "detail/busyheader.hh"
#include "utils/config.hh"
#include "utils/document.hh"
#include "utils/format.hh"
#include "utils/verinfo.h"

namespace cusz {

#if defined(PSZ_USE_CUDA)
const char* BACKEND_TEXT = "cuSZ-Hi";
const char* VERSION_TEXT = "2023-09-05 (unstable)";
const int VERSION = 20230905;
#elif defined(PSZ_USE_HIP)
const char* BACKEND_TEXT = "hipSZ";
const char* VERSION_TEXT = "2023-08-31 (unstable)";
const int VERSION = 20230831;
#elif defined(PSZ_USE_1API)
const char* BACKEND_TEXT = "dpSZ";
const char* VERSION_TEXT = "2023-09-28 (unstable)";
const int VERSION = 20230928;
#endif
const int COMPATIBILITY = 0;
}  // namespace cusz

void capi_psz_version() { printf("\n>>> %s build: %s\n", cusz::BACKEND_TEXT, cusz::VERSION_TEXT); }

void capi_psz_versioninfo()
{
  capi_psz_version();
  printf("\ntoolchain:\n");
  print_CXX_ver();
  print_NVCC_ver();
  printf("\ndriver:\n");
  print_CUDA_driver();
  print_NVIDIA_driver();
  printf("\n");
  CUDA_devices();
}

void pszctx_print_document(bool full_document);
void pszctx_parse_argv(pszctx* ctx, int const argc, char** const argv);
void pszctx_parse_length(pszctx* ctx, const char* lenstr);
void pszctx_parse_length_zyx(pszctx* ctx, const char* lenstr);
void pszctx_parse_control_string(pszctx* ctx, const char* in_str, bool dbg_print);
void pszctx_validate(pszctx* ctx);
void pszctx_load_demo_datasize(pszctx* ctx, void* demodata_name);
void pszctx_set_report(pszctx* ctx, const char* in_str);
void pszctx_set_radius(pszctx* ctx, int _);
// void pszctx_set_huffbyte(pszctx* ctx, int _);
void pszctx_set_huffchunk(pszctx* ctx, int _);
void pszctx_set_densityfactor(pszctx* ctx, int _);

void pszctx_set_report(pszctx* ctx, const char* in_str)
{
  str_list opts;
  psz_helper::parse_strlist(in_str, opts);

  for (auto o : opts) {
    // printf("[psz::dbg::parse] opt: %s\n", o.c_str());
    if (psz_helper::is_kv_pair(o)) {
      auto kv = psz_helper::parse_kv_onoff(o);

      if (kv.first == "cr") ctx->report_cr = kv.second;
      // else if (kv.first == "cr.est")
      //   ctx->report_cr_est = kv.second;
      else if (kv.first == "time")
        ctx->report_time = kv.second;
    }
    else {
      if (o == "cr") ctx->report_cr = true;
      // else if (o == "cr.est")
      //   ctx->report_cr_est = true;
      else if (o == "time")
        ctx->report_time = true;
    }
  }
}

#ifdef PSZ_2505_MERGE
void pszctx_set_datadump(pszctx* ctx, const char* in_str)
{
  str_list opts;
  psz_helper::parse_strlist(in_str, opts);

  for (auto o : opts) {
    if (psz_helper::is_kv_pair(o)) {
      auto kv = psz_helper::parse_kv_onoff(o);

      if (kv.first == "quantcode" or kv.first == "quant")
        ctx->cli->dump_quantcode = kv.second;
      else if (kv.first == "histogram" or kv.first == "hist")
        ctx->cli->dump_hist = kv.second;
      else if (kv.first == "full_huffman_binary" or kv.first == "full_hf")
        ctx->cli->dump_full_hf = kv.second;
    }
    else {
      if (o == "quantcode" or o == "quant")
        ctx->cli->dump_quantcode = true;
      else if (o == "histogram" or o == "hist")
        ctx->cli->dump_hist = true;
      else if (o == "full_huffman_binary" or o == "full_hf")
        ctx->cli->dump_full_hf = true;
    }
  }
}
#endif

/**
 **  >>> syntax
 **  comma-separated key-pairs
 **  "key1=val1,key2=val2[,...]"
 **
 **  >>> example
 **  "predictor=lorenzo,size=3600x1800"
 **
 **/
void pszctx_parse_control_string(pszctx* ctx, const char* in_str, bool dbg_print)
{
  map_t opts;
  psz_helper::parse_strlist_as_kv(in_str, opts);

  if (dbg_print) {
    for (auto kv : opts) printf("%-*s %-s\n", 10, kv.first.c_str(), kv.second.c_str());
    std::cout << "\n";
  }

  std::string k, v;
  char* end;

  auto optmatch = [&](std::vector<std::string> vs) -> bool {
    return psz_utils::check_opt_in_list(k, vs);
  };
  auto is_enabled = [&](auto& v) -> bool { return v == "on" or v == "ON"; };

  for (auto kv : opts) {
    k = kv.first;
    v = kv.second;

    if (optmatch({"type", "dtype"})) {
      psz_utils::check_dtype(v, false);
      ctx->dtype = (v == "f64") or (v == "f8") ? F8 : F4;
    }
    else if (optmatch({"eb", "errorbound"})) {
      ctx->eb = psz_helper::str2fp(v);
    }
    else if (optmatch({"mode"})) {
      psz_utils::check_cuszmode(v);
      ctx->mode = (v == "r2r" or v == "rel") ? Rel : Abs;
    }
    else if (optmatch({"len", "xyz", "dim3"})) {
      pszctx_parse_length(ctx, v.c_str());
    }
    else if (optmatch({"size", "slowest-to-fastest", "zyx"})) {
      pszctx_parse_length_zyx(ctx, v.c_str());
    }
    else if (optmatch({"demo"})) {
      ctx->use_demodata = true;
      strcpy(ctx->demodata_name, v.c_str());
      pszctx_load_demo_datasize(ctx, &v);
    }
    else if (optmatch({"cap", "booklen", "dictsize"})) {
      ctx->dict_size = psz_helper::str2int(v);
      ctx->radius = ctx->dict_size / 2;
    }
    else if (optmatch({"radius"})) {
      ctx->radius = psz_helper::str2int(v);
      ctx->dict_size = ctx->radius * 2;
    }
    else if (optmatch({"huffbyte"})) {
      ctx->huff_bytewidth = psz_helper::str2int(v);
      // ctx->codecs_in_use  = ctx->codec_force_fallback() ? 0b11 /*use both*/
      // : 0b01 /*use 4-byte*/;
    }
    else if (optmatch({"huffchunk"})) {
      ctx->vle_sublen = psz_helper::str2int(v);
      ctx->use_autotune_hf = false;
    }
    else if (optmatch({"predictor"})) {
      strcpy(ctx->dbgstr_pred, v.c_str());

#ifndef PSZ_2505_MERGE
      if (v == "spline" or v == "spline3")
        ctx->pred_type = psz_predtype::Spline;
      else if (v == "lorenzo")
        ctx->pred_type = psz_predtype::Lorenzo;
#else
      if (v == "spline" or v == "spline3" or v == "spl")
        ctx->header->pred_type = psz_predtype::Spline;
      else if (v == "lorenzo" or v == "lrz")
        ctx->header->pred_type = psz_predtype::Lorenzo;
      else if (v == "lorenzo-zigzag" or v == "lrz-zz")
        ctx->header->pred_type = psz_predtype::LorenzoZigZag;
      else if (v == "lorenzo-proto" or v == "lrz-proto")
        ctx->header->pred_type = psz_predtype::LorenzoProto;
#endif
      else {
        printf(
            "[psz::warning::parser] "
            "\"%s\" is not a supported predictor; "
            "fallback to \"lorenzo\".",
            v.c_str());
        ctx->pred_type = psz_predtype::Lorenzo;
      }
    }
#ifdef PSZ_2505_MERGE
    else if (optmatch({"hist", "histogram"})) {
      strcpy(ctx->cli->char_codec1_name, v.c_str());

      if (v == "generic")
        ctx->header->hist_type = psz_histotype::HistogramGeneric;
      else if (v == "sparse")
        ctx->header->hist_type = psz_histotype::HistogramSparse;
    }
    else if (optmatch({"codec", "codec1"})) {
      strcpy(ctx->cli->char_codec1_name, v.c_str());

      if (v == "huffman" or v == "hf")
        ctx->header->codec1_type = psz_codectype::Huffman;
      else if (v == "fzgcodec")
        ctx->header->codec1_type = psz_codectype::FZGPUCodec;
    }
#endif
    else if (optmatch({"gpuverify"}) and is_enabled(v)) {
      ctx->use_gpu_verify = true;
    }
    //// start of Hi configs
    else if (optmatch({"auto_tuning"})) {
      // ctx->intp_param.auto_tuning = psz_helper::str2int(v);
      if (v == "cr-first") { ctx->intp_param.auto_tuning = 3; }
      else if (v == "rd-first") {
        ctx->intp_param.auto_tuning = 6;
      }
      else {
        try {
          ctx->intp_param.auto_tuning = static_cast<uint8_t>(psz_helper::str2int(v));
        }
        catch (...) {
          std::cerr << "[Error] Invalid auto_tuning value: " << v
                    << ". Expected cr-first, rd-first, or an integer.\n";
          exit(1);
        }
      }
    }
    else if (optmatch({"alpha"})) {
      ctx->intp_param.alpha = psz_helper::str2fp(v);
    }
    else if (optmatch({"beta"})) {
      ctx->intp_param.beta = psz_helper::str2fp(v);
    }
    else if (optmatch({"md_0"})) {
      ctx->intp_param.use_md[0] = psz_helper::str2int(v);
    }
    else if (optmatch({"md_1"})) {
      ctx->intp_param.use_md[1] = psz_helper::str2int(v);
    }
    else if (optmatch({"md_2"})) {
      ctx->intp_param.use_md[2] = psz_helper::str2int(v);
    }
    else if (optmatch({"md_3"})) {
      ctx->intp_param.use_md[3] = psz_helper::str2int(v);
    }
    else if (optmatch({"nat_0"})) {
      ctx->intp_param.use_natural[0] = psz_helper::str2int(v);
    }
    else if (optmatch({"nat_1"})) {
      ctx->intp_param.use_natural[1] = psz_helper::str2int(v);
    }
    else if (optmatch({"nat_2"})) {
      ctx->intp_param.use_natural[2] = psz_helper::str2int(v);
    }
    else if (optmatch({"nat_3"})) {
      ctx->intp_param.use_natural[3] = psz_helper::str2int(v);
    }

    else if (optmatch({"rev_0"})) {
      ctx->intp_param.reverse[0] = psz_helper::str2int(v);
    }
    else if (optmatch({"rev_1"})) {
      ctx->intp_param.reverse[1] = psz_helper::str2int(v);
    }
    else if (optmatch({"rev_2"})) {
      ctx->intp_param.reverse[2] = psz_helper::str2int(v);
    }
    else if (optmatch({"rev_3"})) {
      ctx->intp_param.reverse[3] = psz_helper::str2int(v);
    }
    //// end of Hi config
  }
}

void pszctx_create_from_argv(pszctx* ctx, int const argc, char** const argv)
{
  if (argc == 1) {
    pszctx_print_document(false);
    exit(0);
  }

  pszctx_parse_argv(ctx, argc, argv);
  pszctx_validate(ctx);
}

void pszctx_create_from_string(pszctx* ctx, const char* in_str, bool dbg_print)
{
  pszctx_parse_control_string(ctx, in_str, dbg_print);
}

void pszctx_parse_argv(pszctx* ctx, int const argc, char** const argv)
{
  int i = 1;

  auto check_next = [&]() {
    if (i + 1 >= argc) throw std::runtime_error("out-of-range at" + std::string(argv[i]));
  };

  std::string opt;
  auto optmatch = [&](std::vector<std::string> vs) -> bool {
    return psz_utils::check_opt_in_list(opt, vs);
  };

  while (i < argc) {
    if (argv[i][0] == '-') {
      opt = std::string(argv[i]);

      if (optmatch({"-c", "--config"})) {
        check_next();
        pszctx_parse_control_string(ctx, argv[++i], false);
      }
      else if (optmatch({"-R", "--report"})) {
        check_next();
        pszctx_set_report(ctx, argv[++i]);
      }
      else if (optmatch({"-h", "--help"})) {
        pszctx_print_document(true);
        exit(0);
      }
      else if (optmatch({"-v", "--version"})) {
        capi_psz_version();
        exit(0);
      }
      else if (optmatch({"-V", "--versioninfo", "--query-env"})) {
        capi_psz_versioninfo();
        exit(0);
      }
      else if (optmatch({"-m", "--mode"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        ctx->mode = (_ == "r2r" or _ == "rel") ? Rel : Abs;
        if (ctx->mode == Rel) ctx->prep_prescan = true;
      }
      else if (optmatch({"-e", "--eb", "--error-bound"})) {
        check_next();
        char* end;
        ctx->eb = std::strtod(argv[++i], &end);
      }
      else if (optmatch({"-p", "--predictor"})) {
        check_next();
        auto v = std::string(argv[++i]);
        strcpy(ctx->dbgstr_pred, v.c_str());

        if (v == "spline" or v == "spline3") { ctx->pred_type = psz_predtype::Spline; }
        else if (v == "lorenzo") {
          ctx->pred_type = psz_predtype::Lorenzo;
        }
        else {
          printf(
              "[psz::warning::parser] "
              "\"%s\" is not a supported predictor; "
              "fallback to \"lorenzo\".",
              v.c_str());
          ctx->pred_type = psz_predtype::Lorenzo;
        }
      }
      else if (optmatch({"-t", "--type", "--dtype"})) {
        check_next();
        std::string s = std::string(std::string(argv[++i]));
        if (s == "f32" or s == "f4")
          ctx->dtype = F4;
        else if (s == "f64" or s == "f8")
          ctx->dtype = F8;
      }
      else if (optmatch({"-i", "--input"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        strcpy(ctx->infile, _.c_str());
      }
      else if (optmatch({"-l", "--len", "--xyz", "--dim3"})) {
        check_next();
        pszctx_parse_length(ctx, argv[++i]);
      }
      else if (optmatch({"--size", "--zyx", "--slowest-to-fastest"})) {
        check_next();
        pszctx_parse_length_zyx(ctx, argv[++i]);
      }
      else if (optmatch({"-z", "--zip", "--compress"})) {
        ctx->task_construct = true;
      }
      else if (optmatch({"-x", "--unzip", "--decompress"})) {
        ctx->task_reconstruct = true;
      }
      else if (optmatch({"-r", "--dryrun"})) {
        ctx->task_dryrun = true;
      }
      else if (optmatch({"-P", "--pre", "--preprocess"})) {
        check_next();
        std::string pre(argv[++i]);
        if (pre.find("binning") != std::string::npos) { ctx->prep_binning = true; }
      }
      else if (optmatch({"-V", "--verbose"})) {
        ctx->verbose = true;
      }
      else if (optmatch({"--demo"})) {
        check_next();
        ctx->use_demodata = true;
        auto _ = std::string(argv[++i]);
        strcpy(ctx->demodata_name, _.c_str());
        // ctx->demodata_name = std::string(argv[++i]);
        pszctx_load_demo_datasize(ctx, &_);
      }
      else if (optmatch({"-S", "-X", "--skip", "--exclude"})) {
        check_next();
        std::string exclude(argv[++i]);
        if (exclude.find("huffman") != std::string::npos) { ctx->skip_hf = true; }
        if (exclude.find("write2disk") != std::string::npos) { ctx->skip_tofile = true; }
      }
      else if (optmatch({"--opath"})) {
        check_next();
        throw std::runtime_error("[23june] Specifying output path is temporarily disabled.");
        auto _ = std::string(argv[++i]);
        strcpy(ctx->opath, _.c_str());
      }
      else if (optmatch({"--origin", "--compare"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        strcpy(ctx->original_file, _.c_str());
      }
      else if (optmatch({"-a", "--auto"})) {
        check_next();
        // auto _ = std::stoi(argv[++i]);
        // ctx->intp_param.auto_tuning = (uint8_t)_;
        std::string mode = argv[++i];
        if (mode == "cr-first") { ctx->intp_param.auto_tuning = 3; }
        else if (mode == "rd-first") {
          ctx->intp_param.auto_tuning = 6;
        }
        else {
          try {
            ctx->intp_param.auto_tuning = static_cast<uint8_t>(std::stoi(mode));
          }
          catch (...) {
            std::cerr << "[Error] Unknown auto-tuning mode: " << mode
                      << ". Supported: cr-first, rd-first, or an integer value.\n";
            exit(1);
          }
        }
      }
      else if (optmatch({"-s", "--scheme"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "tp") { ctx->use_huffman = false; }
        else if (_ == "cr") {
          ctx->use_huffman = true;
        }
      }

      else if (optmatch({"--sycl-device"})) {
#if defined(PSZ_USE_1API)
        check_next();
        auto _v = string(argv[++i]);
        if (_v == "cpu" or _v == "CPU")
          ctx->device = CPU;
        else if (_v == "gpu" or _v == "GPU")
          ctx->device = INTELGPU;
        else
          ctx->device = INTELGPU;

#else
        throw std::runtime_error(
            "[psz::error] --sycl-device is not supported backend other than "
            "CUDA/HIP.");
#endif
      }
      else {
        const char* notif_prefix = "invalid option value at position ";
        char* notif;
        int size = asprintf(&notif, "%d: %s", i, argv[i]);
        cerr << LOG_ERR << notif_prefix << "\e[1m" << notif << "\e[0m"
             << "\n";
        cerr << std::string(strlen(LOG_NULL) + strlen(notif_prefix), ' ');
        cerr << "\e[1m";
        cerr << std::string(strlen(notif), '~');
        cerr << "\e[0m\n";

        std::cout << LOG_ERR << "Exiting..." << endl;
        exit(-1);
      }
    }
    else {
      const char* notif_prefix = "invalid option at position ";
      char* notif;
      int size = asprintf(&notif, "%d: %s", i, argv[i]);
      cerr << LOG_ERR << notif_prefix << "\e[1m" << notif
           << "\e[0m"
              "\n"
           << std::string(strlen(LOG_NULL) + strlen(notif_prefix), ' ')  //
           << "\e[1m"                                                    //
           << std::string(strlen(notif), '~')                            //
           << "\e[0m\n";

      std::cout << LOG_ERR << "Exiting..." << endl;
      exit(-1);
    }
    i++;
  }
}

void pszctx_load_demo_datasize(pszctx* ctx, void* name)
{
  const std::unordered_map<std::string, std::vector<int>> dataset_entries = {
      {std::string("hacc"), {280953867, 1, 1, 1, 1}},
      {std::string("hacc1b"), {1073726487, 1, 1, 1, 1}},
      {std::string("cesm"), {3600, 1800, 1, 1, 2}},
      {std::string("hurricane"), {500, 500, 100, 1, 3}},
      {std::string("nyx-s"), {512, 512, 512, 1, 3}},
      {std::string("nyx-m"), {1024, 1024, 1024, 1, 3}},
      {std::string("qmc"), {288, 69, 7935, 1, 3}},
      {std::string("qmcpre"), {69, 69, 33120, 1, 3}},
      {std::string("exafel"), {388, 59200, 1, 1, 2}},
      {std::string("rtm"), {235, 849, 849, 1, 3}},
      {std::string("parihaka"), {1168, 1126, 922, 1, 3}}};

  auto demodata_name = *(std::string*)name;

  if (not demodata_name.empty()) {
    auto f = dataset_entries.find(demodata_name);
    if (f == dataset_entries.end()) throw std::runtime_error("no such dataset as" + demodata_name);
    auto demo_xyzw = f->second;

    ctx->x = demo_xyzw[0], ctx->y = demo_xyzw[1], ctx->z = demo_xyzw[2], ctx->w = demo_xyzw[3],
    ctx->ndim = demo_xyzw[4];

    ctx->data_len = ctx->x * ctx->y * ctx->z * ctx->w;
  }
}

#ifndef PSZ_2505_MERGE
void pszctx_parse_length(pszctx* ctx, const char* lenstr)
{
  std::vector<std::string> dims;
  psz_utils::parse_length_literal(lenstr, dims);
  ctx->ndim = dims.size();
  ctx->y = ctx->z = ctx->w = 1;
  ctx->x = psz_helper::str2int(dims[0]);
  if (ctx->ndim >= 2) ctx->y = psz_helper::str2int(dims[1]);
  if (ctx->ndim >= 3) ctx->z = psz_helper::str2int(dims[2]);
  if (ctx->ndim >= 4) ctx->w = psz_helper::str2int(dims[3]);
  ctx->data_len = ctx->x * ctx->y * ctx->z * ctx->w;
}
#else
void pszctx_parse_length(pszctx* ctx, const char* lenstr)
{
  std::vector<std::string> dims;
  psz_utils::parse_length_literal(lenstr, dims);
  ctx->ndim = dims.size();
  ctx->header->y = ctx->header->z = ctx->header->w = 1;
  ctx->header->x = psz_helper::str2int(dims[0]);
  if (ctx->ndim >= 2) ctx->header->y = psz_helper::str2int(dims[1]);
  if (ctx->ndim >= 3) ctx->header->z = psz_helper::str2int(dims[2]);
  if (ctx->ndim >= 4) ctx->header->w = psz_helper::str2int(dims[3]);
  ctx->data_len = ctx->header->x * ctx->header->y * ctx->header->z * ctx->header->w;
}
#endif

#ifndef PSZ_2505_MERGE
void pszctx_parse_length_zyx(pszctx* ctx, const char* lenstr)
{
  std::vector<std::string> dims;
  psz_utils::parse_length_literal(lenstr, dims);
  ctx->ndim = dims.size();
  ctx->y = ctx->z = ctx->w = 1;
  ctx->x = psz_helper::str2int(dims[ctx->ndim - 1]);
  if (ctx->ndim >= 2) ctx->y = psz_helper::str2int(dims[ctx->ndim - 2]);
  if (ctx->ndim >= 3) ctx->z = psz_helper::str2int(dims[ctx->ndim - 3]);
  if (ctx->ndim >= 4) ctx->w = psz_helper::str2int(dims[ctx->ndim - 4]);
  ctx->data_len = ctx->x * ctx->y * ctx->z * ctx->w;
}
#else
void pszctx_parse_length_zyx(pszctx* ctx, const char* lenstr)
{
  std::vector<std::string> dims;
  psz_utils::parse_length_literal(lenstr, dims);
  ctx->ndim = dims.size();
  ctx->header->y = ctx->header->z = ctx->header->w = 1;
  ctx->header->x = psz_helper::str2int(dims[ctx->ndim - 1]);
  if (ctx->ndim >= 2) ctx->header->y = psz_helper::str2int(dims[ctx->ndim - 2]);
  if (ctx->ndim >= 3) ctx->header->z = psz_helper::str2int(dims[ctx->ndim - 3]);
  if (ctx->ndim >= 4) ctx->header->w = psz_helper::str2int(dims[ctx->ndim - 4]);
  ctx->data_len = ctx->header->x * ctx->header->y * ctx->header->z * ctx->header->w;
}
#endif

#ifndef PSZ_2505_MERGE
void pszctx_validate(pszctx* ctx)
{
  bool to_abort = false;
  // if (ctx->infile.empty()) {
  if (ctx->infile[0] == '\0') {
    cerr << LOG_ERR << "must specify input file" << endl;
    to_abort = true;
  }

  if (ctx->data_len == 1 and not ctx->use_demodata) {
    if (ctx->task_construct or ctx->task_dryrun) {
      cerr << LOG_ERR << "wrong input size" << endl;
      to_abort = true;
    }
  }
  if (not ctx->task_construct and not ctx->task_reconstruct and not ctx->task_dryrun) {
    cerr << LOG_ERR << "select compress (-z), decompress (-x) or dryrun (-r)" << endl;
    to_abort = true;
  }
  if (false == psz_utils::check_dtype(ctx->dtype)) {
    if (ctx->task_construct or ctx->task_dryrun) {
      std::cout << ctx->dtype << endl;
      cerr << LOG_ERR << "must specify data type" << endl;
      to_abort = true;
    }
  }
  if (ctx->task_dryrun and ctx->task_construct and ctx->task_reconstruct) {
    cerr << LOG_WARN << "no need to dryrun, compress and decompress at the same time" << endl;
    cerr << LOG_WARN << "dryrun only" << endl << endl;
    ctx->task_construct = false;
    ctx->task_reconstruct = false;
  }
  else if (ctx->task_dryrun and ctx->task_construct) {
    cerr << LOG_WARN << "no need to dryrun and compress at the same time" << endl;
    cerr << LOG_WARN << "dryrun only" << endl << endl;
    ctx->task_construct = false;
  }
  else if (ctx->task_dryrun and ctx->task_reconstruct) {
    cerr << LOG_WARN << "no need to dryrun and decompress at the same time" << endl;
    cerr << LOG_WARN << "will dryrun only" << endl << endl;
    ctx->task_reconstruct = false;
  }

  if (to_abort) {
    pszctx_print_document(false);
    exit(-1);
  }
}
#else
void pszctx_validate(pszctx* ctx)
{
  bool to_abort = false;
  // if (ctx->cli->file_input.empty()) {
  if (ctx->cli->file_input[0] == '\0') {
    cerr << LOG_ERR << "must specify input file" << endl;
    to_abort = true;
  }

  if (not ctx->cli->task_construct and not ctx->cli->task_reconstruct) {
    cerr << LOG_ERR << "select compress (-z) or decompress (-x)." << endl;
    to_abort = true;
  }
  if (false == psz_utils::check_dtype(ctx->header->dtype)) {
    if (ctx->cli->task_construct) {
      std::cout << ctx->header->dtype << endl;
      cerr << LOG_ERR << "must specify data type" << endl;
      to_abort = true;
    }
  }

  if (to_abort) {
    pszctx_print_document(false);
    exit(-1);
  }
}
#endif

#ifndef PSZ_2505_MERGE
void pszctx_print_document(bool full_document)
{
  if (full_document)
    std::cout << "Full document is disabled temporarily." << std::endl;
  else
    std::cout << psz_short_doc << std::endl;
}
#else
void pszctx_print_document(bool full_document)
{
  if (full_document) {
    capi_psz_version();
    std::cout << "\n" << psz_helper::doc_format(psz_full_doc);
  }
  else {
    capi_psz_version();
    std::cout << psz_helper::doc_format(psz_short_doc);
  }
}
#endif

#ifndef PSZ_2505_MERGE
void pszctx_set_rawlen(pszctx* ctx, size_t _x, size_t _y, size_t _z, size_t _w)
{
  ctx->x = _x, ctx->y = _y, ctx->z = _z, ctx->w = _w;

  auto ndim = 4;
  if (ctx->w == 1) ctx->ndim = 3;
  if (ctx->z == 1) ndim = 2;
  if (ctx->y == 1) ndim = 1;

  ctx->ndim = ndim;
  ctx->data_len = ctx->x * ctx->y * ctx->z * ctx->w;

  if (ctx->data_len == 1) throw std::runtime_error("Input data length cannot be 1 (linearized).");
  if (ctx->data_len == 0) throw std::runtime_error("Input data length cannot be 0 (linearized).");
}
#else
void pszctx_set_rawlen(pszctx* ctx, size_t _x, size_t _y, size_t _z)
{
  ctx->header->x = _x, ctx->header->y = _y, ctx->header->z = _z;

  auto ndim = 4;
  if (ctx->header->w == 1) ctx->ndim = 3;
  if (ctx->header->z == 1) ndim = 2;
  if (ctx->header->y == 1) ndim = 1;

  ctx->ndim = ndim;
  ctx->data_len = ctx->header->x * ctx->header->y * ctx->header->z;

  if (ctx->data_len == 1) throw std::runtime_error("Input data length cannot be 1 (linearized).");
  if (ctx->data_len == 0) throw std::runtime_error("Input data length cannot be 0 (linearized).");
}
#endif

#ifndef PSZ_2505_MERGE
void pszctx_set_len(pszctx* ctx, pszlen l) { pszctx_set_rawlen(ctx, l.x, l.y, l.z, l.w); }
#else
void pszctx_set_len(pszctx* ctx, psz_len3 len) { pszctx_set_rawlen(ctx, len.x, len.y, len.z); }
#endif

#ifdef PSZ_2505_MERGE
psz_len3 pszctx_get_len3(pszctx* ctx)
{
  return psz_len3{ctx->header->x, ctx->header->y, ctx->header->z};
}
#endif

void pszctx_set_radius(pszctx* ctx, int _)
{
  ctx->radius = _;
  ctx->dict_size = ctx->radius * 2;
}

#ifdef PSZ_2505_MERGE

psz_len3 pszctx_get_len3(pszctx* ctx)
{
  return psz_len3{ctx->header->x, ctx->header->y, ctx->header->z};
}

void pszctx_set_radius(pszctx* ctx, int _)
{
  ctx->header->radius = _;
  ctx->dict_size = ctx->header->radius * 2;
}

pszctx* pszctx_default_values()
{
  return new pszctx{
      .header =
          new psz_header{
              .dtype = F4,
              .pred_type = DEFAULT_PREDICTOR,
              .hist_type = DEFAULT_HISTOGRAM,
              .codec1_type = DEFAULT_CODEC,
              .mode = Rel,
              .eb = 0.1,
              .radius = 512,
              .vle_sublen = 512,
              .vle_pardeg = -1,
              .x = 1,
              .y = 1,
              .z = 1,
              .w = 1,
              .splen = 0,
          },
      .cli =
          new psz_cli_config{
              .dump_quantcode = false,
              .dump_hist = false,
              .task_construct = false,
              .task_reconstruct = false,
              .rel_range_scan = false,
              .use_gpu_verify = false,
              .skip_tofile = false,
              .skip_hf = false,
              .report_time = false,
              .report_cr = false,
              .verbose = false,
          },
      .dict_size = 1024,
      .data_len = 1,
      .ndim = -1,
      .there_is_memerr = false,
  };
}

void pszctx_set_default_values(pszctx* empty_ctx)
{
  auto default_vals = pszctx_default_values();
  memcpy(empty_ctx, default_vals, sizeof(pszctx));
  delete default_vals;
}

pszctx* pszctx_minimal_workset(
    psz_dtype const dtype, psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec)
{
  auto ws = pszctx_default_values();
  ws->header->dtype = dtype;
  ws->header->pred_type = predictor;
  ws->header->codec1_type = codec;
  ws->dict_size = quantizer_radius * 2;
  ws->header->radius = quantizer_radius;
  return ws;
}

unsigned int CLI_x(psz_arguments* args) { return args->header->x; }
unsigned int CLI_y(psz_arguments* args) { return args->header->y; }
unsigned int CLI_z(psz_arguments* args) { return args->header->z; }
unsigned int CLI_w(psz_arguments* args) { return args->header->w; }
unsigned short CLI_radius(psz_arguments* args) { return args->header->radius; }
unsigned short CLI_bklen(psz_arguments* args) { return args->header->radius * 2; }
psz_dtype CLI_dtype(psz_arguments* args) { return args->header->dtype; }
psz_predtype CLI_predictor(psz_arguments* args) { return args->header->pred_type; }
psz_histotype CLI_hist(psz_arguments* args) { return args->header->hist_type; }
psz_codectype CLI_codec1(psz_arguments* args) { return args->header->codec1_type; }
psz_codectype CLI_codec2(psz_arguments* args) { return args->header->_future_codec2_type; }
psz_mode CLI_mode(psz_arguments* args) { return args->header->mode; }
double CLI_eb(psz_arguments* args) { return args->header->eb; }

#endif