/*
 * fast_mineru._fast_mineru_core -- pybind11 bindings (no Torch dependency)
 *
 * 所有函数收 Python 侧传来的裸 GPU 指针(uintptr_t)。内存分配全在 Python 侧(torch tensor)，
 * C++ 只负责 kernel launch。从 fast_ops 吸收而来，作为 fast_mineru 的一等公民 csrc 层。
 */

#include <cstdint>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include "ocr_preprocess.h"

namespace py = pybind11;
using std::uintptr_t;

// ---- ocr_preprocess_image -----------------------------------------------
void ocr_preprocess_image(
    uintptr_t src_ptr, int src_h, int src_w,
    uintptr_t dst_ptr, int dst_h, int dst_w)
{
    launch_ocr_preprocess(
        reinterpret_cast<const uint8_t*>(src_ptr), src_h, src_w,
        reinterpret_cast<float*>(dst_ptr), dst_h, dst_w);
}

// ---- ocr_preprocess_batch -----------------------------------------------
void ocr_preprocess_batch(
    uintptr_t d_ptrs_ptr, uintptr_t d_hs_ptr, uintptr_t d_ws_ptr,
    uintptr_t batch_out_ptr, int N, int dst_h, int dst_w)
{
    launch_batch_ocr_preprocess(
        reinterpret_cast<const uint8_t* const*>(d_ptrs_ptr),
        reinterpret_cast<const int*>(d_hs_ptr),
        reinterpret_cast<const int*>(d_ws_ptr),
        reinterpret_cast<float*>(batch_out_ptr),
        N, dst_h, dst_w);
}

// ---- ocr_crop_and_bgr ----------------------------------------------------
void ocr_crop_and_bgr(
    uintptr_t full_page_ptr, int page_h, int page_w,
    int crop_x0, int crop_y0, int crop_x1, int crop_y1,
    int paste_x, int paste_y,
    uintptr_t crop_out_ptr, int crop_h, int crop_w)
{
    launch_ocr_crop_and_bgr(
        reinterpret_cast<const uint8_t*>(full_page_ptr), page_h, page_w,
        crop_x0, crop_y0, crop_x1, crop_y1, paste_x, paste_y,
        reinterpret_cast<uint8_t*>(crop_out_ptr), crop_h, crop_w);
}

// ---- ocr_apply_mask ------------------------------------------------------
void ocr_apply_mask(
    uintptr_t img_ptr, int img_h, int img_w,
    uintptr_t mask_boxes_ptr, int num_boxes)
{
    launch_ocr_apply_mask(
        reinterpret_cast<uint8_t*>(img_ptr), img_h, img_w,
        reinterpret_cast<const int*>(mask_boxes_ptr), num_boxes);
}

// ---- ocr_rec_warp --------------------------------------------------------
void ocr_rec_warp(
    uintptr_t src_ptr, int src_h, int src_w,
    uintptr_t M_ptr,
    uintptr_t dst_ptr, int dst_h, int dst_w)
{
    launch_ocr_rec_warp(
        reinterpret_cast<const uint8_t*>(src_ptr), src_h, src_w,
        reinterpret_cast<const float*>(M_ptr),
        reinterpret_cast<uint8_t*>(dst_ptr), dst_h, dst_w);
}

// ---- ocr_rec_resize_norm -------------------------------------------------
void ocr_rec_resize_norm(
    uintptr_t src_ptr, int src_h, int src_w,
    uintptr_t dst_ptr, int resized_w, int imgW)
{
    launch_ocr_rec_resize_norm(
        reinterpret_cast<const uint8_t*>(src_ptr), src_h, src_w,
        reinterpret_cast<float*>(dst_ptr), resized_w, imgW);
}


PYBIND11_MODULE(_fast_mineru_core, m) {
    m.doc() = "fast_mineru core - CUDA-accelerated preprocessing (pointer API)";

    m.def("ocr_preprocess_image", &ocr_preprocess_image,
        py::arg("src_ptr"), py::arg("src_h"), py::arg("src_w"),
        py::arg("dst_ptr"), py::arg("dst_h"), py::arg("dst_w"));

    m.def("ocr_preprocess_batch", &ocr_preprocess_batch,
        py::arg("d_ptrs_ptr"), py::arg("d_hs_ptr"), py::arg("d_ws_ptr"),
        py::arg("batch_out_ptr"), py::arg("N"), py::arg("dst_h"), py::arg("dst_w"));

    m.def("ocr_crop_and_bgr", &ocr_crop_and_bgr,
        py::arg("full_page_ptr"), py::arg("page_h"), py::arg("page_w"),
        py::arg("crop_x0"), py::arg("crop_y0"), py::arg("crop_x1"), py::arg("crop_y1"),
        py::arg("paste_x"), py::arg("paste_y"),
        py::arg("crop_out_ptr"), py::arg("crop_h"), py::arg("crop_w"));

    m.def("ocr_apply_mask", &ocr_apply_mask,
        py::arg("img_ptr"), py::arg("img_h"), py::arg("img_w"),
        py::arg("mask_boxes_ptr"), py::arg("num_boxes"));

    m.def("ocr_rec_warp", &ocr_rec_warp,
        py::arg("src_ptr"), py::arg("src_h"), py::arg("src_w"),
        py::arg("M_ptr"), py::arg("dst_ptr"), py::arg("dst_h"), py::arg("dst_w"));

    m.def("ocr_rec_resize_norm", &ocr_rec_resize_norm,
        py::arg("src_ptr"), py::arg("src_h"), py::arg("src_w"),
        py::arg("dst_ptr"), py::arg("resized_w"), py::arg("imgW"));
}
