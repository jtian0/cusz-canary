#ifndef HUFFMAN_WORKFLOW
#define HUFFMAN_WORKFLOW

/**
 * @file huffman_workflow.cuh
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Workflow of Huffman coding (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
//#include <sys/stat.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "type_trait.hh"

using std::string;

typedef std::tuple<size_t, size_t, size_t, bool> tuple3ul;

template <typename UInt>
double GetEntropyFromFrequency(UInt* freq, size_t len, size_t dict_size = 1024)
{
    double entropy = 0.0;
    for (auto i = 0; i < dict_size; i++) {
        double prob = freq[i] * 1.0 / len;
        entropy += freq[i] != 0 ? -prob * log2(prob) : 0;
    }
    return entropy;
}

namespace draft {

template <typename Huff>
void GatherSpHuffMetadata(size_t* _counts, size_t* d_sp_bits, size_t nchunk, size_t& total_bits, size_t& total_uints);

template <typename Huff>
void ExportCodebook(Huff* d_canon_cb, const string& basename, size_t dict_size);

template <typename Huff>
__global__ void CopyHuffmanUintsDenseToSparse(Huff*, Huff*, size_t*, size_t*, size_t);

template <typename T>
void UseNvcompZip(T* space, size_t& len);

template <typename T>
void UseNvcompUnzip(T** space, size_t& len);

}  // namespace draft

// clang-format off
namespace lossless {

namespace wrapper {
template <typename Input> void GetFrequency(Input*, size_t, unsigned int*, int);

}  // namespace wrapper

namespace interface {

template <typename Quant, typename Huff, typename Data = float>
std::tuple<size_t, size_t, size_t, bool> HuffmanEncode(string& basename, Quant* d_input, size_t len, int dn_chunk, bool to_nvcomp, int dict_size = 1024, bool export_cb=false);

template <typename Quant, typename Huff, typename Data = float>
Quant* HuffmanDecode(std::string& basename, size_t len, int chunk_size, size_t total_uInts, bool nvcomp_in_use, int dict_size = 1024);

template <typename Quant, typename Huff, typename Data = float>
void HuffmanEncodeWithTree_3D(Index<3>::idx_t idx, string& basename, Quant* h_quant_in, size_t len, int dict_size);

}  // namespace interface
}  // namespace lossless

// clang-format on

#endif