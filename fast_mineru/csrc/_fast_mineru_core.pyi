"""fast_mineru core - CUDA-accelerated preprocessing (pointer API)"""
from __future__ import annotations
import fast_mineru.csrc._fast_mineru_core
import typing

__all__ = [
    "ocr_apply_mask",
    "ocr_crop_and_bgr",
    "ocr_preprocess_batch",
    "ocr_preprocess_image",
    "ocr_rec_resize_norm",
    "ocr_rec_warp"
]


def ocr_apply_mask(img_ptr: typing.SupportsInt | typing.SupportsIndex, img_h: typing.SupportsInt | typing.SupportsIndex, img_w: typing.SupportsInt | typing.SupportsIndex, mask_boxes_ptr: typing.SupportsInt | typing.SupportsIndex, num_boxes: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def ocr_crop_and_bgr(full_page_ptr: typing.SupportsInt | typing.SupportsIndex, page_h: typing.SupportsInt | typing.SupportsIndex, page_w: typing.SupportsInt | typing.SupportsIndex, crop_x0: typing.SupportsInt | typing.SupportsIndex, crop_y0: typing.SupportsInt | typing.SupportsIndex, crop_x1: typing.SupportsInt | typing.SupportsIndex, crop_y1: typing.SupportsInt | typing.SupportsIndex, paste_x: typing.SupportsInt | typing.SupportsIndex, paste_y: typing.SupportsInt | typing.SupportsIndex, crop_out_ptr: typing.SupportsInt | typing.SupportsIndex, crop_h: typing.SupportsInt | typing.SupportsIndex, crop_w: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def ocr_preprocess_batch(d_ptrs_ptr: typing.SupportsInt | typing.SupportsIndex, d_hs_ptr: typing.SupportsInt | typing.SupportsIndex, d_ws_ptr: typing.SupportsInt | typing.SupportsIndex, batch_out_ptr: typing.SupportsInt | typing.SupportsIndex, N: typing.SupportsInt | typing.SupportsIndex, dst_h: typing.SupportsInt | typing.SupportsIndex, dst_w: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def ocr_preprocess_image(src_ptr: typing.SupportsInt | typing.SupportsIndex, src_h: typing.SupportsInt | typing.SupportsIndex, src_w: typing.SupportsInt | typing.SupportsIndex, dst_ptr: typing.SupportsInt | typing.SupportsIndex, dst_h: typing.SupportsInt | typing.SupportsIndex, dst_w: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def ocr_rec_resize_norm(src_ptr: typing.SupportsInt | typing.SupportsIndex, src_h: typing.SupportsInt | typing.SupportsIndex, src_w: typing.SupportsInt | typing.SupportsIndex, dst_ptr: typing.SupportsInt | typing.SupportsIndex, resized_w: typing.SupportsInt | typing.SupportsIndex, imgW: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
def ocr_rec_warp(src_ptr: typing.SupportsInt | typing.SupportsIndex, src_h: typing.SupportsInt | typing.SupportsIndex, src_w: typing.SupportsInt | typing.SupportsIndex, M_ptr: typing.SupportsInt | typing.SupportsIndex, dst_ptr: typing.SupportsInt | typing.SupportsIndex, dst_h: typing.SupportsInt | typing.SupportsIndex, dst_w: typing.SupportsInt | typing.SupportsIndex) -> None:
    pass
