/*
 * MFR formula preprocessing kernel — one GPU pass per formula crop.
 *
 * Fuses the UniMERNet pipeline steps that were previously on CPU:
 *   1. Bilinear resize  (longest side → 384, preserve AR, match PIL.Image.BILINEAR resample=2)
 *   2. Center-pad       (normalized white, fill 384×384)
 *   3. Grayscale        (BT.601 luma: 0.114 B + 0.587 G + 0.299 R)
 *   4. Normalize        (mean=0.7931, std=0.1738, scale=1/255)
 *
 * Input:  d_ptrs[i] → BGR uint8 [src_hs[i], src_ws[i], 3]  (GPU, contiguous)
 * Output: d_out       → float32 [N, 1, 384, 384]             (GPU, contiguous)
 *
 * Grid:  (N, 24, 24) blocks of (16, 16, 1) threads = 384×384 threads per image.
 *        One thread per output pixel — zero bank conflicts, coalesced writes.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define OUT_H 384
#define OUT_W 384

// Normalized white (uint8 255 → (1.0-mean)/std) — the pad fill value.
__device__ __constant__ float kPadVal = (1.0f - 0.7931f) / 0.1738f;
__device__ __constant__ float kMean   = 0.7931f;
__device__ __constant__ float kStd    = 0.1738f;
__device__ __constant__ float kScale  = 1.0f / 255.0f;

// ---- device helpers -------------------------------------------------------
__device__ inline float sample_bilinear_gray(
    const uint8_t* src, int src_h, int src_w, float sy, float sx)
{
    // Clamp to valid source range [0, src_h-1), [0, src_w-1)
    sy = fmaxf(0.0f, fminf(sy, (float)(src_h - 1) - 1e-5f));
    sx = fmaxf(0.0f, fminf(sx, (float)(src_w - 1) - 1e-5f));
    int y0 = (int)__float2int_rd(sy);  // floor
    int x0 = (int)__float2int_rd(sx);
    int y1 = min(y0 + 1, src_h - 1);
    int x1 = min(x0 + 1, src_w - 1);
    float wy = sy - (float)y0;
    float wx = sx - (float)x0;

    // BGR packed: src + (y * src_w + x) * 3
    auto load = [&](int row, int col) -> float {
        const uint8_t* p = src + ((size_t)row * src_w + col) * 3;
        // BT.601 luma — identical to cv2.COLOR_BGR2GRAY
        return 0.114f * (float)p[0] + 0.587f * (float)p[1] + 0.299f * (float)p[2];
    };

    float tl = load(y0, x0);
    float tr = load(y0, x1);
    float bl = load(y1, x0);
    float br = load(y1, x1);
    float top    = tl + (tr - tl) * wx;
    float bottom = bl + (br - bl) * wx;
    return top + (bottom - top) * wy;
}

// ---- kernel ---------------------------------------------------------------
__global__ void mfr_preprocess_kernel(
    const uint8_t* const* __restrict__ d_ptrs,
    const int*    __restrict__ d_src_hs,
    const int*    __restrict__ d_src_ws,
    float*        __restrict__ d_out,
    int N)
{
    int n    = blockIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.x + threadIdx.x;
    if (n >= N || out_y >= OUT_H || out_x >= OUT_W) return;

    int src_h = d_src_hs[n];
    int src_w = d_src_ws[n];

    // Resize ratio: longest side → 384, preserve AR.
    float ratio = fminf((float)OUT_H / (float)src_h, (float)OUT_W / (float)src_w);
    int new_h = (int)((float)src_h * ratio + 0.5f);
    int new_w = (int)((float)src_w * ratio + 0.5f);
    int pad_top  = (OUT_H - new_h) / 2;
    int pad_left = (OUT_W - new_w) / 2;

    float val;
    if (out_y < pad_top || out_y >= pad_top + new_h ||
        out_x < pad_left || out_x >= pad_left + new_w)
    {
        val = kPadVal;
    } else {
        // Map output pixel back to source coordinates.
        // ly = out_y - pad_top, lx = out_x - pad_left
        // src_y = ly / ratio,  src_x = lx / ratio
        float src_y = (float)(out_y - pad_top) / ratio;
        float src_x = (float)(out_x - pad_left) / ratio;
        float gray = sample_bilinear_gray(d_ptrs[n], src_h, src_w, src_y, src_x);
        val = (gray * kScale - kMean) / kStd;
    }
    // Output layout: [N, 1, OUT_H, OUT_W] row-major.
    d_out[((size_t)n * OUT_H + out_y) * OUT_W + out_x] = val;
}

// ---- launcher -------------------------------------------------------------
void launch_mfr_preprocess_batch(
    const uint8_t* const* d_ptrs,
    const int*                d_src_hs,
    const int*                d_src_ws,
    float*                    d_out,
    int                       N)
{
    dim3 block(16, 16, 1);
    dim3 grid(N, (OUT_H + 15) / 16, (OUT_W + 15) / 16);
    mfr_preprocess_kernel<<<grid, block>>>(
        d_ptrs, d_src_hs, d_src_ws, d_out, N);
}
