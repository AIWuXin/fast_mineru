"""DBNetTRT —— OCR-det DBNet 的 TensorRT 推理(动态 H/W profile，零拷贝)。

无 monkey-patch：由 pipeline 在构造期把 TextDetector 实例的推理路径绑到本引擎。
输入名 `x`、输出名 `maps`，[B,3,H,W]→[B,1,H,W] fp32。超 profile 范围等比缩放(32 对齐)，
超 batch 拆块。任一块失败返回 None → 调用方回退 torch。
"""
from __future__ import annotations

import torch

from .trt_base import TRTEngine

_DET_PROFILE_MIN = (96, 128)    # (H, W)
_DET_PROFILE_MAX = (736, 960)


class DBNetTRT:
    def __init__(self, engine_path: str,
                 profile_min=_DET_PROFILE_MIN, profile_max=_DET_PROFILE_MAX,
                 max_batch: int = 16, stream: torch.cuda.Stream | None = None):
        self.eng = TRTEngine(engine_path, stream=stream)  # 复用 async_v3 零拷贝
        self.profile_min = profile_min
        self.profile_max = profile_max
        try:
            self.max_batch = self.eng.profile_batch_max("x", 0)
        except Exception:
            self.max_batch = max_batch

    def ensure_bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """超 profile 范围则等比缩放并保持 32 对齐。返回 (tensor, scaled)。复刻 _ensure_trt_bounds。"""
        B, C, H, W = x.shape
        min_h, min_w = self.profile_min
        max_h, max_w = self.profile_max
        if min_h <= H <= max_h and min_w <= W <= max_w:
            return x, False
        scale = min(max_h / H, max_w / W)
        new_h = max(int(round(H * scale / 32)) * 32, 32)
        new_w = max(int(round(W * scale / 32)) * 32, 32)
        while new_h > max_h:
            new_h -= 32
        while new_w > max_w:
            new_w -= 32
        if new_h == H and new_w == W:
            return x, False
        resized = torch.nn.functional.interpolate(
            x, (new_h, new_w), mode="bilinear", align_corners=False)
        return resized, True

    def _infer(self, x: torch.Tensor) -> torch.Tensor | None:
        try:
            o = self.eng.run({"x": x})
        except Exception:
            return None
        return o.get("maps")

    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        """x: [B,3,H,W] → [B,1,H,W] fp32 GPU。超范围缩放，超 batch 拆块，失败整体回退(None)。"""
        x, _scaled = self.ensure_bounds(x)
        B = x.shape[0]
        if B <= self.max_batch:
            return self._infer(x)
        chunks = []
        for i in range(0, B, self.max_batch):
            out = self._infer(x[i:i + self.max_batch])
            if out is None:
                return None
            chunks.append(out)
        return torch.cat(chunks, dim=0)

    def wrap_forward(self, net_forward):
        """包 DBNet net.forward：TRT 出 maps 后包成 {'maps': ...}(与 _build_det_preds 契约一致)。

        注意：net 输入经 _to_inference_dtype 可能是 fp16，TRT I/O 需 fp32；输出 [B,1,H,W]。
        DBNet profile 是动态 H/W，若预处理尺寸超范围 ensure_bounds 会缩放——但那会让 maps 尺寸
        与 shape_list 不符导致后处理框错位，故 wrapper 内**仅在范围内才走 TRT**，超范围回退 torch。
        """
        min_h, min_w = self.profile_min
        max_h, max_w = self.profile_max
        self.hit = 0
        self.miss = 0

        def wrapper(x):
            if x.device.type == "cuda" and x.dim() == 4:
                _, _, H, W = x.shape
                if min_h <= H <= max_h and min_w <= W <= max_w:
                    x_f32 = x.float() if x.dtype != torch.float32 else x
                    B = x_f32.shape[0]
                    out = self._infer(x_f32) if B <= self.max_batch else self.__call__(x_f32)
                    if out is not None:
                        self.hit += 1
                        return {"maps": out}
            self.miss += 1
            return net_forward(x)
        wrapper._fast_mineru_trt = True
        return wrapper
