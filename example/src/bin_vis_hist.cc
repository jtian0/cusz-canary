#include <cstdint>

#include "utils/io.hh"
#include "utils/vis_stat.hh"

int main()
{
  using T = uint16_t;

  auto len = 6480000lu;
  auto data = io::read_binary_to_new_array<T>(
      "/home/jtian/data/cesm_dim3-3600x1800/CLDHGH.f32.1.2e-4.quant_u2", len);

  visualize_histogram<T>("raw hist, 17-bin", data, len, 17);

  visualize_histogram<T>(
      "cropped hist, 17-bin", data, len, 17, false, 512 - 8, 512 + 8);

  return 0;
}