# -*- coding: utf-8 -*-
"""DBNet(OCR-det) + CRNN(OCR-rec) → ONNX 导出 → engines_bin/dbnet.onnx / crnn.onnx。

导出的 ONNX 与引擎完全对齐：
  DBNet:  input "x" [B,3,H,W]  → output "maps"        (dynamic: batch, H, W)
  CRNN:   input "x" [B,3,48,W] → output "ctc_logits"  (dynamic: batch, W)

用法:
  python export_det_rec_onnx.py                 # 默认 lang=ch，导到 engines_bin/
  python export_det_rec_onnx.py --lang ch -o /some/dir
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import _env

OPSET = _env.OPSET


def load_ocr(lang: str):
    """构造 MinerU 的 PytorchPaddleOCR，取出 det/rec 的 torch 子网。"""
    from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR

    print(f"Loading PytorchPaddleOCR(lang={lang!r}) ...")
    ocr = PytorchPaddleOCR(lang=lang)
    det_net = ocr.text_detector.net      # DBNet
    rec_net = ocr.text_recognizer.net    # CRNN
    device = next(det_net.parameters()).device
    det_net.eval()
    rec_net.eval()
    print(f"  det: {type(det_net).__name__}  rec: {type(rec_net).__name__}  device={device}")
    return det_net, rec_net, device


def export_dbnet(net, device, out_dir):
    print("\n=== Exporting DBNet (OCR-det) ===")
    net = net.float()  # 模型权重可能是 fp16，输入统一用 fp32
    dummy = torch.randn(1, 3, 384, 672, device=device)
    path = os.path.join(out_dir, "dbnet.onnx")
    torch.onnx.export(
        net, (dummy,), path,
        input_names=["x"],
        output_names=["maps"],
        dynamic_axes={
            "x":    {0: "batch", 2: "height", 3: "width"},
            "maps": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    with torch.no_grad():
        out = net(dummy)
        out_t = out.get("maps", list(out.values())[0]) if isinstance(out, dict) else out
    print(f"  Saved: {path}")
    print(f"  Verify: in {tuple(dummy.shape)} -> out {tuple(out_t.shape)}")


def export_crnn(net, device, out_dir):
    print("\n=== Exporting CRNN (OCR-rec) ===")
    net = net.float()
    dummy = torch.randn(1, 3, 48, 320, device=device)

    # CRNN forward 返回 {'ctc_logits': tensor, 'ctc_use_raw_logits': bool}，
    # bool 输出 tracer 无法处理。wrap 一下只取 ctc_logits。
    class CRNNWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            out = self.inner(x)
            return out['ctc_logits'] if isinstance(out, dict) else out

    wrapped = CRNNWrapper(net)
    path = os.path.join(out_dir, "crnn.onnx")
    torch.onnx.export(
        wrapped, (dummy,), path,
        input_names=["x"],
        output_names=["ctc_logits"],
        dynamic_axes={
            "x":          {0: "batch", 3: "width"},
            "ctc_logits": {0: "batch", 1: "width"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    with torch.no_grad():
        out = wrapped(dummy)
    print(f"  Saved: {path}")
    print(f"  Verify: in {tuple(dummy.shape)} -> out {tuple(out.shape)}")


def main():
    p = argparse.ArgumentParser(description="Export DBNet + CRNN to ONNX for fast_mineru TRT")
    p.add_argument("--lang", default="ch", help="OCR language (default: ch)")
    p.add_argument("--output", "-o", default=None,
                   help="Output dir (default: engines_bin/)")
    args = p.parse_args()

    out_dir = os.path.abspath(args.output) if args.output else str(_env.ensure_engines_dir())
    os.makedirs(out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    det_net, rec_net, device = load_ocr(args.lang)
    export_dbnet(det_net, device, out_dir)
    export_crnn(rec_net, device, out_dir)

    print(f"\n[OK] dbnet.onnx / crnn.onnx exported to {out_dir}")
    print("Next: build TRT engines with build_engines.py (or invoke build-engines).")


if __name__ == "__main__":
    main()
