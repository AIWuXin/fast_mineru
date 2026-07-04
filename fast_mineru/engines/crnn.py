"""CRNNTRT —— OCR-rec CRNN 的 TensorRT 推理(动态 width，零拷贝)。

无 monkey-patch：由 pipeline 在构造期把 TextRecognizer 的 net.forward 绑到本引擎的 wrapper。
输入名 `x`、输出名 `ctc_logits`，[B,3,48,W]→[B,T,vocab] fp32。width 超上限等比缩放，
超 batch 拆块。任一块失败返回 None → 回退 torch。

wrapper 返回 {'ctc_logits':..., 'ctc_use_raw_logits':True}，与 MinerU CRNN 前向输出契约一致。
"""
from __future__ import annotations

import torch

from .trt_base import TRTEngine

_CRNN_MAX_WIDTH = 2560


class CRNNTRT:
    def __init__(self, engine_path: str, max_batch: int = 16,
                 max_width: int = _CRNN_MAX_WIDTH, stream: torch.cuda.Stream | None = None):
        self.eng = TRTEngine(engine_path, stream=stream)
        self.max_width = max_width
        try:
            self.max_batch = self.eng.profile_batch_max("x", 0)
        except Exception:
            self.max_batch = max_batch

    def _infer(self, x: torch.Tensor) -> torch.Tensor | None:
        try:
            o = self.eng.run({"x": x})
        except Exception:
            return None
        return o.get("ctc_logits")

    def __call__(self, x: torch.Tensor) -> torch.Tensor | None:
        """x: [B,3,48,W] → [B,T,vocab] fp32 GPU。超 batch 拆块，超 width 等比缩放。"""
        B, C, H, W = x.shape
        if B > self.max_batch:
            chunks = []
            for i in range(0, B, self.max_batch):
                out = self._infer(x[i:i + self.max_batch])
                if out is None:
                    return None
                chunks.append(out)
            return torch.cat(chunks, dim=0)
        if W > self.max_width:
            new_w = int(W * (self.max_width / W))
            x = torch.nn.functional.interpolate(
                x, (48, new_w), mode="bilinear", align_corners=False)
        return self._infer(x)

    def wrap_forward(self, net_forward):
        """把 net.forward 包一层：TRT 可用则替换，否则回退原前向。返回新 forward。"""
        self.hit = 0
        self.miss = 0

        def wrapper(x):
            if x.device.type == "cuda":
                x_f32 = x.float() if x.dtype != torch.float32 else x
                trt_out = self(x_f32)
                if trt_out is not None:
                    self.hit += 1
                    return {"ctc_logits": trt_out, "ctc_use_raw_logits": True}
            self.miss += 1
            return net_forward(x)
        wrapper._fast_mineru_trt = True
        return wrapper
