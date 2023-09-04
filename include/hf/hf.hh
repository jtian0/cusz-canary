/**
 * @file codec.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef DAB559E7_A5C1_4342_B17E_17C31DA96EEF
#define DAB559E7_A5C1_4342_B17E_17C31DA96EEF

#include <cstdint>
#include <memory>

#include "cusz/type.h"
#include "hf/hf_struct.h"
#include "mem/memseg_cxx.hh"

namespace cusz {

template <typename E, typename M = u4>
class HuffmanCodec {
 public:
  using BYTE = u1;
  using H = u4;

 private:
  using BOOK = H;
  using SYM = E;

  // TODO shared header
  struct alignas(128) Header {
    static const int HEADER = 0;
    static const int REVBOOK = 1;
    static const int PAR_NBIT = 2;
    static const int PAR_ENTRY = 3;
    static const int BITSTREAM = 4;
    static const int END = 5;

    int self_bytes : 16;
    int booklen : 16;
    int sublen;
    int pardeg;
    size_t original_len;
    size_t total_nbit;
    size_t total_ncell;  // TODO change to uint32_t
    M entry[END + 1];

    M compressed_size() const { return entry[END]; }
  };

  struct runtime_encode_helper {
    static const int TMP = 0;
    static const int FREQ = 1;
    static const int BOOK = 2;
    static const int REVBOOK = 3;
    static const int PAR_NBIT = 4;
    static const int PAR_NCELL = 5;
    static const int PAR_ENTRY = 6;
    static const int BITSTREAM = 7;
    static const int END = 8;

    uint32_t nbyte[END];
  };

  using RTE = runtime_encode_helper;
  using Header = struct Header;

 public:
  // array
  pszmem_cxx<H>* tmp;
  pszmem_cxx<BYTE>* compressed;
  pszmem_cxx<H>* book;
  pszmem_cxx<BYTE>* revbook;
  pszmem_cxx<M>* par_nbit;
  pszmem_cxx<M>* par_ncell;
  pszmem_cxx<M>* par_entry;
  pszmem_cxx<H>* bitstream;

  // helper
  RTE rte;
  // memory
  static const int CELL_BITWIDTH = sizeof(H) * 8;
  // timer
  float _time_book{0.0}, _time_lossless{0.0};

  hf_book* book_desc;
  hf_chunk* chunk_desc_d;
  hf_chunk* chunk_desc_h;
  hf_bitstream* bitstream_desc;

  int pardeg;
  int booklen;
  int numSMs;

 public:
  ~HuffmanCodec();           // dtor
  HuffmanCodec() = default;  // ctor

  // getter
  float time_book() const;
  float time_lossless() const;
  static size_t revbook_bytes(int);
  // getter for internal array
  // H*    expose_book() const;
  // BYTE* expose_revbook() const;

  // compile-time
  constexpr bool can_overlap_input_and_firstphase_encode();
  // public methods
  HuffmanCodec* init(
      size_t const, int const, int const, bool dbg_print = false);
  HuffmanCodec* build_codebook(uint32_t*, int const, void* = nullptr);
  HuffmanCodec* build_codebook(
      pszmem_cxx<uint32_t>*, int const, void* = nullptr);
  HuffmanCodec* encode(E*, size_t const, BYTE**, size_t*, void* = nullptr);
  HuffmanCodec* decode(BYTE*, E*, void* = nullptr, bool = true);
  HuffmanCodec* dump(std::vector<pszmem_dump>, char const*);
  HuffmanCodec* clear_buffer();

 private:
  void hf_merge(
      Header&, size_t const, int const, int const, int const,
      void* stream = nullptr);
  void hf_debug(const std::string, void*, int);
};

}  // namespace cusz

#endif /* DAB559E7_A5C1_4342_B17E_17C31DA96EEF */
