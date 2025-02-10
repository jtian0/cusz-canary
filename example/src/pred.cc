#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include "compbuf.hh"
#include "compressor.hh"
#include "cusz.h"
#include "kernel/lrz/lrz.gpu.hh"
#include "kernel/spv.hh"
#include "mem/cxx_backends.h"
#include "stat/compare.hh"
#include "utils/io.hh"

using std::string;
using namespace psz;
using _portable::utils::fromfile;
using psz::analysis::CPU_probe_extrema;
using psz::analysis::GPU_probe_extrema;

const int radius = 128;

int get_ndim(u4 x, u4 y, u4 z)
{
  if (z == 1 and y == 1)
    return 1;
  else if (z == 1 and y != 1)
    return 2;
  else
    return 3;
}

template <typename T = float>
int run(string fname, size_t x, size_t y, size_t z, double eb, bool use_rel)
{
  printf("1D and 3D.\n");
  auto len = x * y * z;

  auto h_origin = MAKE_UNIQUE_HOST(T, len);
  auto d_origin = MAKE_UNIQUE_DEVICE(T, len);
  auto h_reconst = MAKE_UNIQUE_HOST(T, len);
  auto d_reconst = MAKE_UNIQUE_DEVICE(T, len);

  fromfile(fname, h_origin.get(), len);

  printf("(x, y, z) = (%lu, %lu, %lu)\n", x, y, z);

  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  memcpy_allkinds<H2D>(d_origin.get(), h_origin.get(), len);

  if (use_rel) {
    printf("Using REL mode, input (REL) eb = %f, radius=%d, ", eb, radius);
    double _max_val, _min_val, range;
    {
      CPU_probe_extrema(h_origin.get(), len, _max_val, _min_val, range);
      printf("range (CPU profiled) = %f, ", range);
    }
    {
      GPU_probe_extrema(d_origin.get(), len, _max_val, _min_val, range);
      printf("range (GPU profiled) = %f.", range);
    }
    eb *= range;
    printf("\nrange = %f, adjusted (ABS) eb = %f.\n", range, eb);
  }
  else {
    printf("Using ABS mode, input (ABS) eb = %f, ", eb);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CompressorBufferToggle toggle{
      .err_ctrl_quant = true,
      .compact_outlier = true,
      .anchor = true,
      .histogram = false,
      .compressed = false,
  };

  CompressorBuffer<T> buf(x, y, z, 64, true, &toggle);

  module::GPU_c_lorenzo_nd_with_outlier<T, false>(
      d_origin.get(), {x, y, z}, buf.ectrl(), (void*)buf.outlier(), ebx2, ebx2_r, radius, stream);

  cudaStreamSynchronize(stream);

  float _;
  cout << "compact outliers: " << buf.compact_num_outliers() << endl;
  if (buf.compact_num_outliers() != 0)
    psz::spv_scatter_naive<CUDA>(
        buf.compact_val(), buf.compact_idx(), buf.compact_num_outliers(), d_reconst.get(), &_,
        stream);

  module::GPU_x_lorenzo_nd<T, false>(
      buf.ectrl(), d_reconst.get(), d_reconst.get(), {x, y, z}, ebx2, ebx2_r, radius, stream);

  cudaStreamDestroy(stream);

  auto s = new psz_statistics;
  psz::cuhip::GPU_assess_quality(s, d_origin.get(), d_reconst.get(), x * y * z);
  printf(
      "PSNR\t%lf\t"
      "NRMSE\t%lf\n",
      s->score_PSNR, s->score_NRMSE);

  return 0;
}

template <typename T = float>
int run2d(string fname, size_t x, size_t y, double eb, bool use_rel)
{
  printf("2D pitch-aligned.\n");
  size_t z = 1;
  auto len = x * y * z;
  size_t pitch_T;

  auto h_origin = MAKE_UNIQUE_HOST(T, len);
  auto d_origin = MAKE_UNIQUE_DEVICE_PITCH(T, x, y, pitch_T);
  auto d_origin_nonpitch = MAKE_UNIQUE_DEVICE(T, len);

  printf("pitch_T = %lu\n", pitch_T);

  auto h_reconst = MAKE_UNIQUE_HOST(T, len);
  auto d_reconst = MAKE_UNIQUE_DEVICE_PITCH(T, x, y, pitch_T);
  auto d_reconst_nonpitch = MAKE_UNIQUE_DEVICE(T, len);

  fromfile(fname, h_origin.get(), len);

  printf("(x, y, z) = (%lu, %lu, %lu)\n", x, y, z);

  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  memcpy_allkinds<H2D>(d_origin_nonpitch.get(), h_origin.get(), len);
  cudaMemcpy2D(
      d_origin.get(), pitch_T, h_origin.get(), x * sizeof(T), x * sizeof(T), y,
      cudaMemcpyHostToDevice);

  // if (use_rel) {
  //   printf("Using REL mode, input (REL) eb = %f, radius=%d, ", eb, radius);
  //   double _max_val, _min_val, range;
  //   {
  //     CPU_probe_extrema(h_origin.get(), len, _max_val, _min_val, range);
  //     printf("range (CPU profiled) = %f, ", range);
  //   }
  //   {
  //     GPU_probe_extrema(d_origin.get(), len, _max_val, _min_val, range);
  //     printf("range (GPU profiled) = %f.", range);
  //   }
  //   eb *= range;
  //   printf("\nrange = %f, adjusted (ABS) eb = %f.\n", range, eb);
  // }
  // else {
  //   printf("Using ABS mode, input (ABS) eb = %f, ", eb);
  // }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CompressorBufferToggle toggle{
      .err_ctrl_quant = true,
      .compact_outlier = true,
      .anchor = true,
      .histogram = false,
      .compressed = false,
  };

  CompressorBuffer<T> buf(x, y, z, 64, true, &toggle);

  module::GPU_c_lorenzo_nd_with_outlier<T, false>(
      d_origin.get(), {x, y, z}, buf.ectrl(), (void*)buf.outlier(), ebx2, ebx2_r, radius, stream,
      pitch_T, buf.pitch_Eq);

  cudaStreamSynchronize(stream);

  float _;
  cout << "compact outliers: " << buf.compact_num_outliers() << endl;
  if (buf.compact_num_outliers() != 0)
    psz::spv_scatter_naive<CUDA>(
        buf.compact_val(), buf.compact_idx(), buf.compact_num_outliers(), d_reconst.get(), &_,
        stream);

  module::GPU_x_lorenzo_nd<T, false>(
      buf.ectrl(), d_reconst.get(), d_reconst.get(), {x, y, z}, ebx2, ebx2_r, radius, stream,
      pitch_T, buf.pitch_Eq);

  cudaStreamDestroy(stream);

  auto s = new psz_statistics;
  cudaMemcpy2D(
      d_reconst_nonpitch.get(), x * sizeof(T), d_reconst.get(), pitch_T, x * sizeof(T), y,
      cudaMemcpyDeviceToDevice);
  psz::cuhip::GPU_assess_quality(s, d_origin_nonpitch.get(), d_reconst_nonpitch.get(), x * y * z);
  printf(
      "PSNR\t%lf\t"
      "NRMSE\t%lf\n",
      s->score_PSNR, s->score_NRMSE);

  return 0;
}

int main(int argc, char** argv)
{
  if (argc < 6) {
    printf(
        "0     1              2     3  4  5  6   7\n"
        "PROG  /path/to/data  dtype X  Y  Z  eb  use_rel\n");
    exit(1);
  }
  else {
    auto fname = std::string(argv[1]);
    auto dtype = std::string(argv[2]);
    auto x = atoi(argv[3]);
    auto y = atoi(argv[4]);
    auto z = atoi(argv[5]);
    auto eb = std::stod(argv[6]);
    auto const use_rel = std::string(argv[7]) == "yes";

    auto ndim = get_ndim(x, y, z);
    printf("ndim = %d\n", ndim);

    if (dtype == "f4" or dtype == "f32") {
      if (get_ndim(x, y, z) == 2)
        return run2d<float>(fname, x, y, eb, use_rel);
      else
        return run<float>(fname, x, y, z, eb, use_rel);
    }
    else if (dtype == "f8" or dtype == "f64") {
      if (get_ndim(x, y, z) == 2)
        run2d<double>(fname, x, y, eb, use_rel);
      else
        return run<double>(fname, x, y, z, eb, use_rel);
    }
    else
      return -1;
  }
}