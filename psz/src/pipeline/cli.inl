/**
 * @file cli.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-02-20
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CLI_CUH
#define CLI_CUH

#include "cusz.h"
#include "cusz/header.h"
#include "cusz/type.h"
#include "detail/busyheader.hh"
#include "detail/port.hh"
//
#include "cusz/context.h"
#include "lc_gen/lc_gen.h"
#include "mem.hh"
#include "tehm.hh"
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
#include "utils/analyzer.hh"
#endif
#include "utils/err.hh"
#include "utils/query.hh"
#include "utils/viewer.hh"

namespace cusz {

template <typename Data = float>
class CLI {
 private:
  using T = Data;

 public:
  CLI() = default;

 private:
  void write_compressed_to_disk(
      std::string compressed_name, uint8_t* compressed, size_t compressed_len)
  {
    auto file = new memobj<uint8_t>(compressed_len, 1, 1, "cusza");
    file->dptr(compressed)->control({MallocHost, D2H})->file(compressed_name.c_str(), ToFile);
    // ->control({FreeHost});

    delete file;
  }

  // template <typename compressor_t>
  void do_construct(pszctx* ctx, psz_compressor* compressor, void* stream)
  {
    auto input = new memobj<T>(ctx->x, ctx->y, ctx->z, "uncompressed");

    uint8_t* compressed;
    size_t compressed_len;
    pszheader header;

    input->control({MallocHost, Malloc})->file(ctx->infile, FromFile)->control({H2D});

    // adjust eb
    if (ctx->mode == Rel) {
      ctx->rel_eb = ctx->eb;
      double _1, _2, rng;
      input->extrema_scan(_1, _2, rng);
      ctx->eb *= rng;
    }

    psz::TimeRecord timerecord;

    pszlen uncomp_len = pszlen{ctx->x, ctx->y, ctx->z, 1};

    psz_compress_init(compressor, uncomp_len, ctx);

    psz_compress(
        compressor, input->dptr(), uncomp_len, &compressed, &compressed_len, &header,
        (void*)&timerecord, stream);

    printf("\n(c) COMPRESSION REPORT\n");

    if (ctx->report_time) psz::TimeRecordViewer::view_timerecord(&timerecord, &header);
    if (ctx->report_cr) psz::TimeRecordViewer::view_cr(&header);

    write_compressed_to_disk(std::string(ctx->infile) + ".cusza", compressed, compressed_len);

    delete input;
  }

  // template <typename compressor_t>
  void do_reconstruct(pszctx* ctx, psz_compressor* compressor, void* stream)
  {
    // extract basename w/o suffix
    auto basename = std::string(ctx->infile);
    basename = basename.substr(0, basename.rfind('.'));

    // all lengths in metadata
    auto compressed_len = psz_utils::filesize(ctx->infile);

    auto compressed = new memobj<uint8_t>(compressed_len, 1, 1, "compressed");

    compressed->control({MallocHost, Malloc})->file(ctx->infile, FromFile)->control({H2D});

    auto header = new psz_header;
    memcpy(header, compressed->hptr(), sizeof(psz_header));
    auto len = psz_utils::uncompressed_len(header);
    ctx->use_huffman = header->with_huffman;

    auto decompressed = new memobj<T>(len, 1, 1, "decompressed");
    auto outlier_tmp = new memobj<T>(len, 1, 1, "outlier_tmp");
    decompressed->control({MallocHost, Malloc});
    outlier_tmp->control({MallocHost, Malloc});

    auto original = new memobj<T>(len, 1, 1, "original-cmp");

    psz::TimeRecord timerecord;

    pszlen decomp_len = pszlen{header->x, header->y, header->z, 1};

    psz_decompress_init(compressor, header);
    // compressor->header->intp_param = ctx->intp_param;

    psz_decompress(
        compressor, compressed->dptr(), psz_utils::filesize(header), decompressed->dptr(),
        outlier_tmp->dptr(), decomp_len, (void*)&timerecord, stream);

    if (ctx->report_time)
      psz::TimeRecordViewer::view_decompression(&timerecord, decompressed->bytes());
    psz::view(header, decompressed, original, ctx->original_file);

    if (not ctx->skip_tofile)
      decompressed->control({D2H})->file(std::string(basename + ".cuszx").c_str(), ToFile);

    // decompressed->control({FreeHost, Free});
    delete decompressed;
    delete original;
  }

 public:
  // TODO determine dtype & predictor in here
  void dispatch(pszctx* ctx)
  {
    // TODO disable predictor selection; to specify in another way
    // auto predictor = ctx->predictor;

    psz_framework* framework = pszdefault_framework();
    psz_compressor* compressor = psz_create(framework, F4);

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    cudaStream_t stream;
    CHECK_GPU(cudaStreamCreate(&stream));

    // TODO enable f8
    if (ctx->task_construct) do_construct(ctx, compressor, stream);
    if (ctx->task_reconstruct) do_reconstruct(ctx, compressor, stream);
    if (stream) cudaStreamDestroy(stream);

#elif defined(PSZ_USE_1API)

    sycl::queue q;
    auto plist = sycl::property_list(
        sycl::property::queue::in_order(), sycl::property::queue::enable_profiling());

    if (ctx->device == CPU)
      q = sycl::queue(sycl::cpu_selector_v, plist);
    else if (ctx->device == INTELGPU)
      q = sycl::queue(sycl::gpu_selector_v, plist);
    else
      q = sycl::queue(sycl::default_selector_v, plist);

    // TODO enable f8
    if (ctx->task_construct) do_construct(ctx, compressor, &q);
    if (ctx->task_reconstruct) do_reconstruct(ctx, compressor, &q);

#endif

    delete framework;
    delete compressor;
  }
};

}  // namespace cusz

#endif
