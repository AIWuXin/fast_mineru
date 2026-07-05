#pragma once
#include <cstdint>

/// Launch mfr_preprocess_batch kernel.
/// Input: d_ptrs[i] points to uint8 BGR [src_hs[i], src_ws[i], 3] on device.
/// Output: [N, 1, 384, 384] float32 on device.
/// Each formula crop is resized (longest side→384, bilinear, preserve AR),
/// center-padded with normalized white, then grayscale+normalize(mean/std).
void launch_mfr_preprocess_batch(
    const std::uint8_t* const* d_ptrs,
    const int*                d_src_hs,
    const int*                d_src_ws,
    float*                    d_out,
    int                       N);
