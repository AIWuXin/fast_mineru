"""MFREncoderTRT —— pp_formulanet 编码器(PPHGNetV2/DonutSwin backbone)的 TensorRT 前向。

MFR 段近 90% 耗时在 net(encoder+decoder)。decoder 已两段式 TRT；encoder 是卷积、单次前向、
输入固定 [B,1,384,384]、无动态尺寸 —— 最适合 TRT(fast_ops 实测约占公式识别 ~52% 耗时)。

注入方式：包 backbone.forward，TRT 出 last_hidden_state 包成 DonutSwinModelOutput，decoder 仍走
torch。构造期一次绑定(不猴子补丁 singleton，与 MFR-decoder 注入同构)。引擎 I/O：
  输入  pixel_values     [B,1,384,384] float32(导出图内部自带 1→3 通道 repeat)
  输出  last_hidden_state [B,144,2048] float32
B 超 max_batch 自动拆块；任一块失败整体回退 torch。
"""
from __future__ import annotations

import torch

from .trt_base import TRTEngine


class MFREncoderTRT:
    """封装 encoder 引擎，提供 backbone.forward 替身。"""

    _IN = "pixel_values"
    _OUT = "last_hidden_state"

    def __init__(self, engine_path: str, output_cls, max_batch: int | None = None,
                 debug: bool = False):
        self.engine = TRTEngine(engine_path)
        self.output_cls = output_cls
        self.debug = debug
        try:
            self.max_batch = max_batch or self.engine.profile_batch_max(self._IN)
        except Exception:
            self.max_batch = max_batch or 16
        self._hit = 0
        self._miss = 0

    def _infer(self, x: torch.Tensor) -> torch.Tensor | None:
        try:
            outs = self.engine.run({self._IN: x})
            return outs[self._OUT]
        except Exception:
            return None

    def _run_trt(self, x: torch.Tensor) -> torch.Tensor | None:
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

    def make_wrapper(self, orig_forward):
        """返回替换 backbone.forward 的函数。非 cuda tensor / TRT 失败 → 回退 orig_forward。"""
        def wrapper(x):
            if isinstance(x, torch.Tensor) and x.is_cuda:
                x_f32 = x.float().contiguous() if x.dtype != torch.float32 else x.contiguous()
                trt_out = self._run_trt(x_f32)
                if trt_out is not None:
                    self._hit += 1
                    return self.output_cls(
                        last_hidden_state=trt_out,
                        pooler_output=None,
                        hidden_states=None,
                        attentions=False,
                        reshaped_hidden_states=None,
                    )
            self._miss += 1
            return orig_forward(x)
        wrapper._fast_mineru_trt = True
        return wrapper

    @property
    def hit(self):
        return self._hit

    @property
    def miss(self):
        return self._miss
