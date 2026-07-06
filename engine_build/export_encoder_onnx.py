# -*- coding: utf-8 -*-
"""导出 pp_formulanet_plus_m 的 ENCODER(PPHGNetV2_B6_Formula) → engines_bin/encoder_ppformulanet.onnx。

实测 I/O:
  输入 : [B, 1, 384, 384] float32   (单通道！前处理已转灰度)
  输出 : last_hidden_state [B, 144, 2048] float32

在 CPU 上导出（避免 GPU 特定算子污染图，TRT 自会在 GPU 优化）。
路径全部经 _env 定位，无硬编码。
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

import _env

os.environ.setdefault("MINERU_FORMULA_CH_SUPPORT", "True")

MFR_DIR = _env.get_mfr_weight_dir()
OUT_ONNX = _env.onnx_path("encoder_ppformulanet.onnx")
OPSET = _env.OPSET

device = "cpu"
print("导出设备 =", device, " opset =", OPSET)

from mineru.model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer

rec = FormulaRecognizer(MFR_DIR, device)
net = rec.net
net.eval()


class EncoderWrapper(torch.nn.Module):
    """只暴露 backbone，输出裁剪为单个 last_hidden_state 张量，便于 ONNX/TRT。"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            return out["last_hidden_state"]
        return out


enc = EncoderWrapper(net.backbone).to(device).eval()

# 真实前处理造 dummy 输入 [1,1,384,384]
img = np.array(Image.fromarray((np.arange(120 * 400).reshape(120, 400) % 255).astype("uint8")).convert("RGB"))
batch = rec.pre_tfs["UniMERNetImgDecode"](imgs=[img])
batch = rec.pre_tfs["UniMERNetTestTransform"](imgs=batch)
batch = rec.pre_tfs["LatexImageFormat"](imgs=batch)
inp = rec.pre_tfs["ToBatch"](imgs=batch)
dummy = torch.from_numpy(inp[0]).to(device).float()
print("dummy 输入 shape =", tuple(dummy.shape), " dtype =", dummy.dtype)
assert dummy.shape[1:] == (1, 384, 384), f"意外的输入通道/尺寸: {tuple(dummy.shape)}"

with torch.no_grad():
    ref = enc(dummy)
print("torch encoder 输出 shape =", tuple(ref.shape), " dtype =", ref.dtype)

torch.onnx.export(
    enc,
    dummy,
    OUT_ONNX,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "last_hidden_state": {0: "batch"},
    },
    opset_version=OPSET,
    do_constant_folding=True,
)
print("✓ 已导出:", OUT_ONNX)

# ---- 数值自检: onnxruntime vs torch ----
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
    o1 = sess.run(None, {"pixel_values": dummy.numpy()})[0]
    d1 = np.abs(o1 - ref.numpy())
    print(f"[单batch] onnx vs torch  max|Δ|={d1.max():.3e}  mean|Δ|={d1.mean():.3e}")
    d3 = torch.cat([dummy, dummy * 0.9, dummy * 1.1], 0)
    with torch.no_grad():
        r3 = enc(d3).numpy()
    o3 = sess.run(None, {"pixel_values": d3.numpy()})[0]
    dd = np.abs(o3 - r3)
    print(f"[batch=3] onnx vs torch  max|Δ|={dd.max():.3e}  mean|Δ|={dd.mean():.3e}  out_shape={o3.shape}")
    ok = d1.max() < 1e-3 and dd.max() < 1e-3
    print("数值对齐:", "✓ 通过 (max|Δ|<1e-3)" if ok else "⚠ 偏大，需检查")
except ImportError:
    print("onnxruntime 未安装，跳过自检。")
