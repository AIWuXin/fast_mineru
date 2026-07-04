"""fast_mineru.csrc — CUDA 预处理算子(pybind11)。

编译产物 `_fast_mineru_core.pyd` 无 Torch 依赖：所有 GPU 张量操作在 Python 侧用 torch 完成，
C++ 只收裸指针(uintptr_t)做 kernel launch。本模块负责 Windows DLL 加载修复 + 类型化封装。
"""
import os
import ctypes
import glob

import torch

# ── Windows DLL 加载修复 ─────────────────────────────────────────────
# 先加载 torch DLL 与 CUDA runtime，再导入 .pyd，避免 "DLL load failed"。
_this_dir = os.path.dirname(os.path.abspath(__file__))

try:
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
except Exception:
    _torch_lib = None
if _torch_lib and os.path.isdir(_torch_lib):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
    for _dll in ("torch_python.dll", "torch_cpu.dll", "torch_cuda.dll", "c10.dll"):
        _p = os.path.join(_torch_lib, _dll)
        if os.path.exists(_p):
            try:
                ctypes.CDLL(_p)
            except Exception:
                pass


def _add_cuda_dll_dir():
    """CUDA runtime DLL 目录：包内自带 → 环境变量 → 常见安装路径。"""
    if hasattr(os, "add_dll_directory") and glob.glob(os.path.join(_this_dir, "cudart64_*.dll")):
        try:
            os.add_dll_directory(_this_dir)  # POST_BUILD 拷进来的 cudart
        except OSError:
            pass
    cands = []
    for _var in ("CUDA_PATH", "CUDA_HOME"):
        _v = os.environ.get(_var)
        if _v:
            cands.append(os.path.join(_v, "bin"))
    cands += sorted(glob.glob(
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin"), reverse=True)
    cands += ["/usr/local/cuda/lib64", "/usr/local/cuda/bin"]
    for _d in cands:
        if _d and os.path.isdir(_d) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass


_add_cuda_dll_dir()

from . import _fast_mineru_core as _core  # noqa: E402


# ════════════════════════════════════════════════════════════════════
#  类型化封装(内部调用裸指针 C++ API)
# ════════════════════════════════════════════════════════════════════

def ocr_preprocess_image(src: torch.Tensor, dst_h: int, dst_w: int) -> torch.Tensor:
    """单图 OCR-det 预处理。src: GPU uint8 [H,W,3] → float32 [3,dst_h,dst_w]。"""
    dst = torch.empty(3, dst_h, dst_w, dtype=torch.float32, device=src.device)
    _core.ocr_preprocess_image(src.data_ptr(), src.size(0), src.size(1),
                               dst.data_ptr(), dst_h, dst_w)
    return dst


def ocr_preprocess_batch(images, dst_h, dst_w, d_ptrs=None, d_hs=None, d_ws=None):
    """批量 OCR-det 预处理。返回 (batch_out, d_ptrs, d_hs, d_ws)，metadata buffer 可复用。"""
    N = len(images)
    device = images[0].device
    if d_ptrs is None or d_ptrs.size(0) < N:
        d_ptrs = torch.empty(N, dtype=torch.int64, device=device)
    if d_hs is None or d_hs.size(0) < N:
        d_hs = torch.empty(N, dtype=torch.int32, device=device)
    if d_ws is None or d_ws.size(0) < N:
        d_ws = torch.empty(N, dtype=torch.int32, device=device)
    d_ptrs[:N].copy_(torch.tensor([im.data_ptr() for im in images], dtype=torch.int64))
    d_hs[:N].copy_(torch.tensor([im.size(0) for im in images], dtype=torch.int32))
    d_ws[:N].copy_(torch.tensor([im.size(1) for im in images], dtype=torch.int32))
    batch_out = torch.empty(N, 3, dst_h, dst_w, dtype=torch.float32, device=device)
    _core.ocr_preprocess_batch(d_ptrs.data_ptr(), d_hs.data_ptr(), d_ws.data_ptr(),
                               batch_out.data_ptr(), N, dst_h, dst_w)
    return batch_out, d_ptrs, d_hs, d_ws


def ocr_crop_and_bgr(full_page, crop_x0, crop_y0, crop_x1, crop_y1, paste_x=50, paste_y=50):
    """整页裁一块 + 白 padding + RGB→BGR。full_page: GPU uint8[H,W,3] → BGR uint8。"""
    crop_h = (crop_y1 - crop_y0) + paste_y * 2
    crop_w = (crop_x1 - crop_x0) + paste_x * 2
    crop_out = torch.empty(crop_h, crop_w, 3, dtype=torch.uint8, device=full_page.device)
    _core.ocr_crop_and_bgr(full_page.data_ptr(), full_page.size(0), full_page.size(1),
                           crop_x0, crop_y0, crop_x1, crop_y1, paste_x, paste_y,
                           crop_out.data_ptr(), crop_h, crop_w)
    return crop_out


def ocr_apply_mask(img: torch.Tensor, mask_boxes: torch.Tensor):
    """公式区域遮罩(原位)。img: GPU uint8[H,W,3]；mask_boxes: int32[N,4]。"""
    if mask_boxes.device.type != "cuda":
        mask_boxes = mask_boxes.to(device=img.device, dtype=torch.int32)
    if mask_boxes.dtype != torch.int32:
        mask_boxes = mask_boxes.to(dtype=torch.int32)
    _core.ocr_apply_mask(img.data_ptr(), img.size(0), img.size(1),
                         mask_boxes.data_ptr(), mask_boxes.size(0))


def ocr_rec_warp(src: torch.Tensor, M: torch.Tensor, dst_h: int, dst_w: int) -> torch.Tensor:
    """透视变换裁切文本行。src: GPU uint8[H,W,3] BGR；M: float32[9] → BGR uint8。"""
    if M.device.type != "cuda":
        M = M.to(device=src.device, dtype=torch.float32)
    dst = torch.empty(dst_h, dst_w, 3, dtype=torch.uint8, device=src.device)
    _core.ocr_rec_warp(src.data_ptr(), src.size(0), src.size(1),
                       M.data_ptr(), dst.data_ptr(), dst_h, dst_w)
    return dst


def ocr_rec_resize_norm(src: torch.Tensor, resized_w: int, imgW: int) -> torch.Tensor:
    """resize + normalize(/127.5-1) + pad。src: GPU uint8[H,W,3] BGR → float32[3,48,imgW]。"""
    dst = torch.empty(3, 48, imgW, dtype=torch.float32, device=src.device)
    _core.ocr_rec_resize_norm(src.data_ptr(), src.size(0), src.size(1),
                              dst.data_ptr(), resized_w, imgW)
    return dst


__all__ = [
    "ocr_preprocess_image", "ocr_preprocess_batch", "ocr_crop_and_bgr",
    "ocr_apply_mask", "ocr_rec_warp", "ocr_rec_resize_norm",
]
