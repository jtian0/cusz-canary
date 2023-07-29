/**
 * @file _canonical.cuh
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-10
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef B684F0FA_8869_4DDF_9467_2E28E967AC06
#define B684F0FA_8869_4DDF_9467_2E28E967AC06

#include <cstdint>
#include <cstring>

#include "cusz/type.h"

static constexpr int HFSERIAL_CANON_CPU = 0;
static constexpr int HFSERIAL_CANON_CUDA = 1;

template <typename E = uint32_t, typename H = uint32_t>
class hf_canon_space {
 private:
  H *_icb, *_ocb, *_canon;
  int *_numl, *_iterby, *_first, *_entry;
  E* _keys;

 public:
  // public var
  uint16_t const booklen;
  static const auto TYPE_BITS = sizeof(H) * 8;

  // public fn
  hf_canon_space(uint16_t booklen) : booklen(booklen) { init(); }
  ~hf_canon_space()
  {
    // delete[] _icb,
    delete[] _ocb, delete[] _canon;
    delete[] _keys;
    delete[] _numl, delete[] _iterby, delete[] _first, delete[] _entry;
  }
  void init()
  {
    // booklen
    // _icb = new H[booklen], memset(_icb, 0, sizeof(H) * booklen);
    _ocb = new H[booklen], memset(_ocb, 0, sizeof(H) * booklen);
    _canon = new H[booklen], memset(_canon, 0, sizeof(H) * booklen);
    _keys = new E[booklen], memset(_keys, 0, sizeof(E) * booklen);
    // type bits
    _numl = new int[TYPE_BITS], memset(_numl, 0, sizeof(int) * TYPE_BITS);
    _iterby = new int[TYPE_BITS], memset(_iterby, 0, sizeof(int) * TYPE_BITS);
    _first = new int[TYPE_BITS], memset(_first, 0, sizeof(int) * TYPE_BITS);
    _entry = new int[TYPE_BITS], memset(_entry, 0, sizeof(int) * TYPE_BITS);
  }

  // accessor
  H*& icb() { return _icb; }
  H* ocb() { return _ocb; }
  H* canon() { return _canon; }
  E* keys() { return _keys; }
  int* numl() { return _numl; }
  int* iterby() { return _iterby; }
  int* first() { return _first; }
  int* entry() { return _entry; }

  H& icb(int i) { return _icb[i]; }
  H& ocb(int i) { return _ocb[i]; }
  H& canon(int i) { return _canon[i]; }
  E& keys(int i) { return _keys[i]; }
  int& numl(int i) { return _numl[i]; }
  int& iterby(int i) { return _iterby[i]; }
  int& first(int i) { return _first[i]; }
  int& entry(int i) { return _entry[i]; }
  // run
  int canonize();
};

template <typename E, typename H>
int canonize_on_gpu(uint8_t* binary_in, uint32_t booklen, void* stream);

// TODO: switch to execution policy
// template <typename E, typename H>
// int hf_canonize_serial(void* s);

#endif /* B684F0FA_8869_4DDF_9467_2E28E967AC06 */
