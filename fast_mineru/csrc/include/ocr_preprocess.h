/*
 * OCR-det 预处理融合 kernel
 *
 * 融合 DetResizeForTest + NormalizeImage + ToCHWImage 三步为单一 GPU kernel。
 * 输入:  RGB uint8 HWC, GPU tensor data_ptr
 * 输出:  CHW float32,   GPU tensor data_ptr
 * 无内存分配 —— 所有 buffer 由 PyTorch 预先分配。
 */

#ifndef OCR_PREPROCESS_CUH
#define OCR_PREPROCESS_CUH

// ── 前向声明（pybind11 可见） ───────────────────────────────────────

/*
 * 对一张图的 OCR-det 预处理：resize + normalize + HWC→CHW。
 *
 * src      : GPU 上 BGR uint8 [H, W, 3]
 * src_h, src_w : 源图尺寸
 * dst      : GPU 上 float32 [3, dst_h, dst_w]  （输出 CHW）
 * dst_h, dst_w: 目标尺寸（由 DetResizeForTest 决定）
 * stream   : CUDA stream（默认 0）
 */
void launch_ocr_preprocess(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int dst_h, int dst_w,
    cudaStream_t stream = 0
);

/*
 * 批量版本：每张图独立预处理后放入连续 batch buffer。
 *
 * images      : GPU 上指针对数组，每个指向 RGB uint8 [H, W, 3]
 * batch_out   : GPU 上 float32 [N, 3, dst_h, dst_w]
 * N, dst_h, dst_w
 */
void launch_batch_ocr_preprocess(
    const uint8_t* const* images,
    const int* src_hs, const int* src_ws,
    float* batch_out,
    int N, int dst_h, int dst_w,
    cudaStream_t stream = 0
);

/*
 * 从整页图裁剪一块 + 白色 padding + RGB→BGR 转换。
 *
 * full_page   : GPU 上 RGB uint8 [page_h, page_w, 3]
 * crop_x0..y1 : 裁剪区域（整页坐标）
 * paste_x/y   : padding 偏移
 * crop_out    : GPU 上 BGR uint8 [crop_h, crop_w, 3]（输出 buffer）
 * crop_h/w    : 输出尺寸 = (crop_x1-crop_x0 + 2*paste_x, ...)
 */
void launch_ocr_crop_and_bgr(
    const uint8_t* full_page, int page_h, int page_w,
    int crop_x0, int crop_y0, int crop_x1, int crop_y1,
    int paste_x, int paste_y,
    uint8_t* crop_out, int crop_h, int crop_w,
    cudaStream_t stream = 0
);

/*
 * 对 BGR uint8 图应用公式区域遮罩（置白）。
 *
 * img        : GPU 上 BGR uint8 [img_h, img_w, 3]，原位修改
 * mask_boxes : GPU 上 int [num_boxes, 4]，每行 [x0, y0, x1, y1]
 * num_boxes  : mask_boxes 行数
 */
void launch_ocr_apply_mask(
    uint8_t* img, int img_h, int img_w,
    const int* mask_boxes, int num_boxes,
    cudaStream_t stream = 0
);

/*
 * OCR-rec: 透视变换裁切文本行。匹配 OpenCV warpPerspective + INTER_LINEAR + BORDER_REPLICATE。
 *
 * src    : GPU 上 BGR uint8 [src_h, src_w, 3]（源图）
 * M      : GPU 上 float[9]（3x3 逆透视矩阵，dst->src）
 * dst    : GPU 上 BGR uint8 [dst_h, dst_w, 3]（输出裁切）
 */
void launch_ocr_rec_warp(
    const uint8_t* src, int src_h, int src_w,
    const float* M,
    uint8_t* dst, int dst_h, int dst_w,
    cudaStream_t stream = 0
);

/*
 * OCR-rec: resize + normalize(/127.5-1) + pad。
 * 输入 uint8 [src_h, src_w, 3] BGR，
 * 输出 float32 [3, 48, imgW]，先 resize 到 (resized_w, 48) 再 pad 到 imgW。
 */
void launch_ocr_rec_resize_norm(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int resized_w, int imgW,
    cudaStream_t stream = 0
);

#endif // OCR_PREPROCESS_CUH
