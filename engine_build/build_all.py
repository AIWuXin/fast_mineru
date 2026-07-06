# -*- coding: utf-8 -*-
"""一键从零构建 fast_mineru 的全部 TensorRT 引擎。

顺序：
  0) 架构自检 —— 换原版 mineru==3.4.1 的最大风险点：pp_formulanet/OCR 结构若与导出假设不符，
     这里明确报错并中止，避免产出「能构建但推理错」的坏引擎。
  1) 导出全部 ONNX（各 export_*.py，独立子进程，避免 CUDA context/模型加载互相干扰）。
  2) trtexec 编译全部 engine（build_engines.py）。
  3) 产物落 engines_bin/（config.py::resolve() 默认读这里）。

用法:
  python engine_build/build_all.py                 # 全流程（crnn tf32）
  python engine_build/build_all.py --crnn both     # crnn 同时 tf32+fp16
  python engine_build/build_all.py --skip-export    # 只编译（onnx 已在）
"""
import argparse
import io
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _env

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _run_script(name, *extra):
    """子进程跑 engine_build/ 下的脚本，实时透传输出。失败即抛。"""
    script = os.path.join(HERE, name)
    print(f"\n{'='*70}\n▶ {name} {' '.join(extra)}\n{'='*70}", flush=True)
    rc = subprocess.run([PY, script, *extra]).returncode
    if rc != 0:
        raise SystemExit(f"✗ {name} 失败 (rc={rc})，中止构建。")


def self_check():
    """架构一致性自检：真实 forward 验证 shape 契约（比检查类名更鲁棒）。"""
    print(f"\n{'='*70}\n▶ 架构自检（pp_formulanet + OCR 是否符合导出假设）\n{'='*70}", flush=True)
    os.environ.setdefault("MINERU_FORMULA_CH_SUPPORT", "True")
    import numpy as np
    import torch
    from PIL import Image

    mfr_dir = _env.get_mfr_weight_dir()
    from mineru.model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer
    rec = FormulaRecognizer(mfr_dir, "cuda")
    net = rec.net; net.eval()
    head = net.head

    # encoder：真实前处理 → backbone，断言输出 [1, CROSS, *]
    img = np.array(Image.fromarray(((np.arange(80 * 300).reshape(80, 300)) % 255).astype("uint8")).convert("RGB"))
    b = rec.pre_tfs["UniMERNetImgDecode"](imgs=[img])
    b = rec.pre_tfs["UniMERNetTestTransform"](imgs=b)
    b = rec.pre_tfs["LatexImageFormat"](imgs=b)
    inp = torch.from_numpy(rec.pre_tfs["ToBatch"](imgs=b)[0]).to("cuda").float()
    assert inp.shape[1:] == (1, 384, 384), f"encoder 输入通道/尺寸不符: {tuple(inp.shape)}"
    with torch.no_grad():
        enc = net.backbone(inp)["last_hidden_state"]
    assert enc.shape[1] == _env.CROSS, f"encoder 输出序列长度应={_env.CROSS}, 实为 {enc.shape[1]}"
    print(f"  ✓ encoder: 输入 {tuple(inp.shape)} → last_hidden_state {tuple(enc.shape)}")

    # decoder：首步 present 数应 = N_LAYERS*4
    if head.config_decoder.hidden_size != head.encoder_hidden_size:
        enc = head.enc_to_dec_proj(enc)
    bos = torch.tensor([[0]], device="cuda")
    with torch.no_grad():
        out1 = head.decoder(input_ids=bos, attention_mask=torch.ones_like(bos),
                            encoder_hidden_states=enc, encoder_attention_mask=None,
                            past_key_values=None, use_cache=True, return_dict=True,
                            output_attentions=False, output_hidden_states=False)
    n_present = sum(len(layer) for layer in out1.past_key_values)
    assert n_present == _env.N_LAYERS * 4, f"decoder present 数应={_env.N_LAYERS*4}, 实为 {n_present}"
    print(f"  ✓ decoder: {len(out1.past_key_values)} 层 × 4 = {n_present} present（N_LAYERS={_env.N_LAYERS}）")

    # OCR：det/rec 子网存在
    from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
    ocr = PytorchPaddleOCR(lang="ch")
    assert getattr(ocr.text_detector, "net", None) is not None, "OCR text_detector.net 缺失"
    assert getattr(ocr.text_recognizer, "net", None) is not None, "OCR text_recognizer.net 缺失"
    print(f"  ✓ OCR: det={type(ocr.text_detector.net).__name__} rec={type(ocr.text_recognizer.net).__name__}")

    # 释放，避免占显存影响后续子进程
    del rec, net, ocr
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("  ✓ 架构自检通过 —— 原版 mineru==3.4.1 与导出假设一致")


def main():
    p = argparse.ArgumentParser(description="One-click build of all fast_mineru TRT engines")
    p.add_argument("--crnn", default="tf32", choices=["tf32", "fp16", "both"])
    p.add_argument("--skip-export", action="store_true", help="跳过 ONNX 导出，仅 trtexec 编译")
    p.add_argument("--skip-selfcheck", action="store_true", help="跳过架构自检（不推荐）")
    args = p.parse_args()

    if not args.skip_selfcheck:
        self_check()

    if not args.skip_export:
        _run_script("export_encoder_onnx.py")
        _run_script("export_decoder_init_onnx.py")
        _run_script("export_decoder_onnx.py")
        _run_script("export_det_rec_onnx.py")

    _run_script("build_engines.py", "--crnn", args.crnn)

    print(f"\n{'='*70}\n✓ 全部引擎构建完成 → {_env.ENGINES_DIR}\n{'='*70}")
    print("现在可运行: uv run fast-mineru <pdf>")


if __name__ == "__main__":
    main()
