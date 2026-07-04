#include <cuda_runtime.h>
#include "ocr_preprocess.h"

// Normalization constants (from MinerU's NormalizeImage config)
#define MEAN_R 0.485f
#define MEAN_G 0.456f
#define MEAN_B 0.406f
#define STD_R  0.229f
#define STD_G  0.224f
#define STD_B  0.225f

// Per-pixel helper: bilinear sample + normalize + write CHW
__device__ inline void process_pixel(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int dst_h, int dst_w,
    int x, int y, int ch)
{
    float sx = (x + 0.5f) * src_w / dst_w - 0.5f;
    float sy = (y + 0.5f) * src_h / dst_h - 0.5f;

    sx = fmaxf(0.0f, fminf(sx, (float)(src_w - 1)));
    sy = fmaxf(0.0f, fminf(sy, (float)(src_h - 1)));

    int ix = (int)sx;
    int iy = (int)sy;
    float dx = sx - (float)ix;
    float dy = sy - (float)iy;

    int ix1 = min(ix + 1, src_w - 1);
    int iy1 = min(iy + 1, src_h - 1);

    float p00 = (float)src[iy  * src_w * 3 + ix  * 3 + ch];
    float p10 = (float)src[iy  * src_w * 3 + ix1 * 3 + ch];
    float p01 = (float)src[iy1 * src_w * 3 + ix  * 3 + ch];
    float p11 = (float)src[iy1 * src_w * 3 + ix1 * 3 + ch];

    float top = p00 + (p10 - p00) * dx;
    float bot = p01 + (p11 - p01) * dx;
    float val = top + (bot - top) * dy;

    float mean = (ch == 0) ? MEAN_R : (ch == 1) ? MEAN_G : MEAN_B;
    float stdv = (ch == 0) ? STD_R  : (ch == 1) ? STD_G  : STD_B;
    val = (val / 255.0f - mean) / stdv;

    dst[ch * dst_h * dst_w + y * dst_w + x] = val;
}

// Single image kernel
__global__ void ocr_preprocess_kernel(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int dst_h, int dst_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_h * dst_w * 3) return;

    int ch = idx % 3;
    int p = idx / 3;
    int y = p / dst_w;
    int x = p % dst_w;
    process_pixel(src, src_h, src_w, dst, dst_h, dst_w, x, y, ch);
}

// Batch kernel
__global__ void batch_kernel(
    const uint8_t* const* images,
    const int* src_hs, const int* src_ws,
    float* batch_out,
    int N, int dst_h, int dst_w)
{
    int per_img = dst_h * dst_w * 3;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * per_img) return;

    int img_id = idx / per_img;
    int local = idx % per_img;

    int ch = local % 3;
    int p = local / 3;
    int y = p / dst_w;
    int x = p % dst_w;

    const uint8_t* src = images[img_id];
    int src_h = src_hs[img_id];
    int src_w = src_ws[img_id];
    float* dst = batch_out + img_id * per_img;

    process_pixel(src, src_h, src_w, dst, dst_h, dst_w, x, y, ch);
}

// Crop from full page + white padding + RGB->BGR
__global__ void ocr_crop_and_bgr_kernel(
    const uint8_t* full_page, int page_h, int page_w,
    int crop_x0, int crop_y0, int crop_x1, int crop_y1,
    int paste_x, int paste_y,
    uint8_t* crop_out, int crop_h, int crop_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= crop_h * crop_w * 3) return;

    int ch = idx % 3;
    int p = idx / 3;
    int y = p / crop_w;
    int x = p % crop_w;

    int crop_w_src = crop_x1 - crop_x0;
    int crop_h_src = crop_y1 - crop_y0;

    if (x < paste_x || x >= paste_x + crop_w_src ||
        y < paste_y || y >= paste_y + crop_h_src) {
        crop_out[idx] = 255;
        return;
    }

    int src_x = x - paste_x + crop_x0;
    int src_y = y - paste_y + crop_y0;

    if (src_x < 0 || src_x >= page_w || src_y < 0 || src_y >= page_h) {
        crop_out[idx] = 255;
        return;
    }

    int page_off = src_y * page_w * 3 + src_x * 3;
    switch (ch) {
        case 0: crop_out[idx] = full_page[page_off + 2]; break;
        case 1: crop_out[idx] = full_page[page_off + 1]; break;
        case 2: crop_out[idx] = full_page[page_off + 0]; break;
    }
}

// Apply formula mask (set masked pixels to white)
__global__ void ocr_apply_mask_kernel(
    uint8_t* img, int img_h, int img_w,
    const int* mask_boxes, int num_boxes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= img_h * img_w) return;

    int y = idx / img_w;
    int x = idx % img_w;

    for (int m = 0; m < num_boxes; m++) {
        int x0 = mask_boxes[m * 4 + 0];
        int y0 = mask_boxes[m * 4 + 1];
        int x1 = mask_boxes[m * 4 + 2];
        int y1 = mask_boxes[m * 4 + 3];
        if (x >= x0 && x < x1 && y >= y0 && y < y1) {
            int off = y * img_w * 3 + x * 3;
            img[off + 0] = 255;
            img[off + 1] = 255;
            img[off + 2] = 255;
            return;
        }
    }
}

// OCR-rec: perspective warp (matches OpenCV warpPerspective + INTER_LINEAR + BORDER_REPLICATE)
// M is 3x3 inverse perspective matrix (dst->src), pre-computed on CPU.
__global__ void ocr_rec_warp_kernel(
    const uint8_t* src, int src_h, int src_w,
    const float* M,
    uint8_t* dst, int dst_h, int dst_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_h * dst_w * 3) return;

    int ch = idx % 3;
    int p = idx / 3;
    int y = p / dst_w;
    int x = p % dst_w;

    float sx = M[0] * x + M[1] * y + M[2];
    float sy = M[3] * x + M[4] * y + M[5];
    float sz = M[6] * x + M[7] * y + M[8];
    sx /= sz;
    sy /= sz;

    sx = fmaxf(0.0f, fminf(sx, (float)(src_w - 1)));
    sy = fmaxf(0.0f, fminf(sy, (float)(src_h - 1)));

    int ix = (int)sx;
    int iy = (int)sy;
    float dx = sx - (float)ix;
    float dy = sy - (float)iy;
    int ix1 = min(ix + 1, src_w - 1);
    int iy1 = min(iy + 1, src_h - 1);

    float v = (1.0f - dy) * (1.0f - dx) * (float)src[iy  * src_w * 3 + ix  * 3 + ch]
            + (1.0f - dy) * dx         * (float)src[iy  * src_w * 3 + ix1 * 3 + ch]
            + dy         * (1.0f - dx) * (float)src[iy1 * src_w * 3 + ix  * 3 + ch]
            + dy         * dx          * (float)src[iy1 * src_w * 3 + ix1 * 3 + ch];

    dst[y * dst_w * 3 + x * 3 + ch] = (uint8_t)fminf(255.0f, fmaxf(0.0f, v));
}

// OCR-rec: resize + normalize(/127.5-1) + pad
// Input uint8 [src_h, src_w, 3] BGR, output float32 [3, 48, imgW].
// Resize to (resized_w, 48) then pad to imgW, matching resize_norm_img.
__global__ void ocr_rec_resize_norm_kernel(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int resized_w, int imgW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 48 * imgW * 3) return;

    int ch = idx % 3;
    int p = idx / 3;
    int y = p / imgW;
    int x = p % imgW;

    if (x >= resized_w) {
        dst[ch * 48 * imgW + y * imgW + x] = 0.0f;
        return;
    }

    float sx = (x + 0.5f) * src_w / (float)resized_w - 0.5f;
    float sy = (y + 0.5f) * src_h / 48.0f - 0.5f;
    sx = fmaxf(0.0f, fminf(sx, (float)(src_w - 1)));
    sy = fmaxf(0.0f, fminf(sy, (float)(src_h - 1)));

    int ix = (int)sx;
    int iy = (int)sy;
    float dx = sx - (float)ix;
    float dy = sy - (float)iy;
    int ix1 = min(ix + 1, src_w - 1);
    int iy1 = min(iy + 1, src_h - 1);

    float v = (1.0f - dy) * (1.0f - dx) * (float)src[iy  * src_w * 3 + ix  * 3 + ch]
            + (1.0f - dy) * dx         * (float)src[iy  * src_w * 3 + ix1 * 3 + ch]
            + dy         * (1.0f - dx) * (float)src[iy1 * src_w * 3 + ix  * 3 + ch]
            + dy         * dx          * (float)src[iy1 * src_w * 3 + ix1 * 3 + ch];

    dst[ch * 48 * imgW + y * imgW + x] = v / 127.5f - 1.0f;
}


// Host entry points

void launch_ocr_preprocess(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int dst_h, int dst_w,
    cudaStream_t stream)
{
    int total = dst_h * dst_w * 3;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    ocr_preprocess_kernel<<<grid, block, 0, stream>>>(src, src_h, src_w, dst, dst_h, dst_w);
}

void launch_batch_ocr_preprocess(
    const uint8_t* const* images,
    const int* src_hs, const int* src_ws,
    float* batch_out,
    int N, int dst_h, int dst_w,
    cudaStream_t stream)
{
    int total = N * dst_h * dst_w * 3;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    batch_kernel<<<grid, block, 0, stream>>>(images, src_hs, src_ws, batch_out, N, dst_h, dst_w);
}

void launch_ocr_crop_and_bgr(
    const uint8_t* full_page, int page_h, int page_w,
    int crop_x0, int crop_y0, int crop_x1, int crop_y1,
    int paste_x, int paste_y,
    uint8_t* crop_out, int crop_h, int crop_w,
    cudaStream_t stream)
{
    int total = crop_h * crop_w * 3;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    ocr_crop_and_bgr_kernel<<<grid, block, 0, stream>>>(
        full_page, page_h, page_w,
        crop_x0, crop_y0, crop_x1, crop_y1,
        paste_x, paste_y,
        crop_out, crop_h, crop_w
    );
}

void launch_ocr_apply_mask(
    uint8_t* img, int img_h, int img_w,
    const int* mask_boxes, int num_boxes,
    cudaStream_t stream)
{
    int total = img_h * img_w;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    ocr_apply_mask_kernel<<<grid, block, 0, stream>>>(
        img, img_h, img_w, mask_boxes, num_boxes
    );
}

void launch_ocr_rec_warp(
    const uint8_t* src, int src_h, int src_w,
    const float* M,
    uint8_t* dst, int dst_h, int dst_w,
    cudaStream_t stream)
{
    int total = dst_h * dst_w * 3;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    ocr_rec_warp_kernel<<<grid, block, 0, stream>>>(src, src_h, src_w, M, dst, dst_h, dst_w);
}

void launch_ocr_rec_resize_norm(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int resized_w, int imgW,
    cudaStream_t stream)
{
    int total = 48 * imgW * 3;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    ocr_rec_resize_norm_kernel<<<grid, block, 0, stream>>>(src, src_h, src_w, dst, resized_w, imgW);
}
