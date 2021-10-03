/**
 * @file predictor.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_INCLUDE_PREDICTOR_HH
#define CUSZ_INCLUDE_PREDICTOR_HH

#include <cstdint>

namespace cusz {

template <typename T, typename E>
class PredictorAbstraction {
   private:
    void partition_workspace();

   public:
    // helper functions
    virtual uint32_t get_workspace_nbyte() const = 0;
    virtual uint32_t get_quant_len() const       = 0;
    virtual uint32_t get_anchor_len() const      = 0;
    virtual float    get_time_elapsed() const    = 0;

    // "real" methods
    virtual ~PredictorAbstraction()      = default;
    virtual void construct(T*, T*, E*)   = 0;
    virtual void reconstruct(T*, E*, T*) = 0;
};

}  // namespace cusz

#endif