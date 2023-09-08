/**
 * @file cli.cuh
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

#include "busyheader.hh"
#include "cusz.h"
#include "cusz/type.h"
#include "header.h"
#include "port.hh"
//
#include "context.h"
#include "dryrun.hh"
#include "mem.hh"
#include "tehm.hh"
#include "utils/analyzer.hh"
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

  template <typename T>
  static void do_dryrun(pszctx* ctx, bool dualquant = true)
  {
    GpuStreamT stream;
    GpuStreamCreate(&stream);

    auto x = ctx->x, y = ctx->y, z = ctx->z;
    auto eb = ctx->eb;
    auto r2r = ctx->mode == Rel;
    auto fname = ctx->infile;

    pszmem_cxx<T>* original = new pszmem_cxx<T>(x, y, z, "original");
    pszmem_cxx<T>* reconst = new pszmem_cxx<T>(x, y, z, "reconst");
    original->control({MallocHost, Malloc});
    reconst->control({MallocHost, Malloc});

    double max, min, rng;
    auto len = original->len();

    original->debug();

    original->file(fname, FromFile)->control({ASYNC_H2D}, stream);
    CHECK_GPU(GpuStreamSync((GpuStreamT)stream));

    if (r2r) original->extrema_scan(max, min, rng), eb *= rng;

    psz::cuda_hip_compat::dryrun(
        len, original->dptr(), reconst->dptr(), eb, stream);

    reconst->control({D2H});

    cusz_stats stat;
    psz::assess_quality<CPU>(&stat, reconst->hptr(), original->hptr(), len);
    psz::print_metrics_cross<T>(&stat, 0, true);

    // destroy
    original->control({FreeHost, Free});
    reconst->control({FreeHost, Free});

    delete original;
    delete reconst;

    GpuStreamDestroy(stream);
  }

 private:
  void write_compressed_to_disk(
      std::string compressed_name, uint8_t* compressed, size_t compressed_len)
  {
    auto file = new pszmem_cxx<uint8_t>(compressed_len, 1, 1, "cusza");
    file->dptr(compressed)
        ->control({MallocHost, D2H})
        ->file(compressed_name.c_str(), ToFile);
    // ->control({FreeHost});

    delete file;
  }

  // template <typename compressor_t>
  void do_construct(
      pszctx* ctx, cusz_compressor* compressor, GpuStreamT stream)
  {
    auto input = new pszmem_cxx<T>(ctx->x, ctx->y, ctx->z, "uncompressed");

    uint8_t* compressed;
    size_t compressed_len;
    pszheader header;

    input->control({MallocHost, Malloc})
        ->file(ctx->infile, FromFile)
        ->control({H2D});

    // adjust eb
    if (ctx->mode == Rel) {
      double _1, _2, rng;
      input->extrema_scan(_1, _2, rng);
      ctx->eb *= rng;
    }

    TimeRecord timerecord;

    // pszrc* config = new pszrc{
    //     .eb = ctx->eb,
    //     .mode = Rel,
    //     .pred_type = ctx->pred_type,
    //     .est_cr = ctx->report_cr_est};
    pszlen uncomp_len = pszlen{ctx->x, ctx->y, ctx->z, 1};

    psz_compress_init(compressor, uncomp_len, ctx);

    psz_compress(
        compressor, input->dptr(), uncomp_len, &compressed, &compressed_len,
        &header, (void*)&timerecord, stream);

    if (ctx->report_time)
      TimeRecordViewer::view_compression(
          &timerecord, input->m->bytes, compressed_len);
    write_compressed_to_disk(
        std::string(ctx->infile) + ".cusza", compressed, compressed_len);

    delete input;
  }

  // template <typename compressor_t>
  void do_reconstruct(
      pszctx* ctx, cusz_compressor* compressor, GpuStreamT stream)
  {
    // extract basename w/o suffix
    auto basename = std::string(ctx->infile);
    basename = basename.substr(0, basename.rfind('.'));

    // all lengths in metadata
    auto compressed_len = psz_utils::filesize(ctx->infile);

    auto compressed =
        new pszmem_cxx<uint8_t>(compressed_len, 1, 1, "compressed");

    compressed->control({MallocHost, Malloc})
        ->file(ctx->infile, FromFile)
        ->control({H2D});

    auto header = new cusz_header;
    memcpy(header, compressed->hptr(), sizeof(cusz_header));
    auto len = psz_utils::uncompressed_len(header);

    auto decompressed = new pszmem_cxx<T>(len, 1, 1, "decompressed");
    decompressed->control({MallocHost, Malloc});

    auto original = new pszmem_cxx<T>(len, 1, 1, "original-cmp");

    TimeRecord timerecord;

    pszlen decomp_len = pszlen{header->x, header->y, header->z, 1};

    psz_decompress_init(compressor, header);
    psz_decompress(
        compressor, compressed->dptr(), psz_utils::filesize(header),
        decompressed->dptr(), decomp_len, (void*)&timerecord, stream);

    if (ctx->report_time)
      TimeRecordViewer::view_decompression(
          &timerecord, decompressed->m->bytes);
    psz::view(header, decompressed, original, ctx->original_file);

    if (not ctx->skip_tofile)
      decompressed->control({D2H})->file(
          std::string(basename + ".cuszx").c_str(), ToFile);

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

    cusz_framework* framework = pszdefault_framework();
    cusz_compressor* compressor = cusz_create(framework, F4);

    GpuStreamT stream;
    CHECK_GPU(GpuStreamCreate(&stream));

    // TODO enable f8
    if (ctx->task_dryrun) do_dryrun<float>(ctx);
    if (ctx->task_construct) do_construct(ctx, compressor, stream);
    if (ctx->task_reconstruct) do_reconstruct(ctx, compressor, stream);

    if (stream) GpuStreamDestroy(stream);
  }
};

}  // namespace cusz

#endif
