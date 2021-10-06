/**
 * @file binding.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-10-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_BINDING_HH
#define CUSZ_BINDING_HH

// NVCC requires the two headers; clang++ does not.
#include <limits>
#include <type_traits>

/**
 * ------------
 * default path
 * ------------
 *
 * Predictor<T, E, (FP)>
 *           |  |   ^
 *           v  |   |
 * SpReducer<T> |   +---- default "fast-lowlowprecision"
 *              v
 *      Encoder<E, H>
 */

template <class Predictor, class SpReducer, class Encoder>
struct PredictorReducerEncoderBinding {
    using T1 = typename Predictor::Origin;
    using T2 = typename Predictor::Anchor;
    using E1 = typename Predictor::ErrCtrl;
    using T3 = typename SpReducer::Origin;  // SpRecuder -> BYTE, omit
    using E2 = typename Encoder::Origin;
    using H  = typename Encoder::Encoded;

    using PREDICTOR = Predictor;
    using SPREDUCER = SpReducer;
    using ENCODER   = Encoder;

    static void type_matching()
    {
        static_assert(
            std::is_same<T1, T2>::value and std::is_same<T1, T3>::value,
            "Predictor::Origin, Predictor::Anchor, and SpReducer::Origin must be the same.");
        static_assert(std::is_same<E1, E2>::value, "Predictor::ErrCtrl and Encoder::Origin must be the same.");

        // TODO this is the restriction for now.
        static_assert(std::is_floating_point<T1>::value, "Predictor::Origin must be floating-point type.");

        // TODO open up the possibility of (E1 neq E2) and (E1 being FP)
        static_assert(
            std::numeric_limits<E1>::is_integer and std::is_unsigned<E1>::value,
            "Predictor::ErrCtrl must be unsigned integer.");

        static_assert(
            std::numeric_limits<H>::is_integer and std::is_unsigned<H>::value,
            "Encoder::Encoded must be unsigned integer.");
    }
};

/**
 * -------------
 * sp-aware path
 * -------------
 *
 * Predictor<T, E, (FP)>
 *              |
 *              v
 *    SpReducer<E>
 */

template <class Predictor, class SpReducer>
struct PredictorReducerBinding {
    using T1 = typename Predictor::Origin;
    using T2 = typename Predictor::Anchor;
    using E1 = typename Predictor::ErrCtrl;
    using E2 = typename SpReducer::Origin;

    using PREDICTOR = Predictor;
    using SPREDUCER = SpReducer;

    // SpRecuder -> BYTE, omit

    static void type_matching()
    {
        static_assert(std::is_same<T1, T2>::value, "Predictor::Origin and Predictor::Anchor must be the same.");

        // alternatively, change Output of Predictor in place of Origin
        static_assert(std::is_same<E1, E2>::value, "Predictor::ErrCtrl and SpReducer::Origin must be the same.");

        // TODO this is the restriction for now.
        static_assert(std::is_floating_point<T1>::value, "Predictor::Origin must be floating-point type.");

        // TODO this is the restriction for now.
        static_assert(std::is_floating_point<E1>::value, "Predictor::ErrCtrl must be floating-point type.");
    }
};

#endif