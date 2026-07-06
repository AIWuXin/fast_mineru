# -*- coding: utf-8 -*-
"""用 trtexec 把 engines_bin/ 下的 ONNX 编译成 5 个 TensorRT 引擎（路径无关、可移植）。

引擎与 fast_mineru/config.py::resolve() 的默认名严格对齐：
  decoder_init.onnx        → decoder_init_fp16.engine        (--fp16, batch 动态)
  decoder_with_past.onnx   → decoder_with_past_fp16.engine   (--fp16, batch × past_len 双动态)
  encoder_ppformulanet.onnx→ encoder_ppformulanet_fp16.engine(--fp16, batch 动态)
  dbnet.onnx               → dbnet.engine                    (--fp16, batch/H/W 动态)
  crnn.onnx                → crnn.engine (tf32) / crnn_fp16.engine (--fp16)

★ 动态 shape 的 min/opt/max 是引擎正确性 + 性能曲线的关键，逐字保留自原始构建脚本
  (build_decoder_engines.py) 与 fast_ops/README.md 的验证参数，请勿随意改动。

用法:
  python build_engines.py                         # 构建全部（crnn 默认 tf32）
  python build_engines.py --only encoder,dbnet    # 只构建指定引擎
  python build_engines.py --crnn both             # crnn 同时构建 tf32 + fp16
"""
import argparse
import io
import os
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _env

N_LAYERS = _env.N_LAYERS       # 6
CROSS = _env.CROSS             # 144
# decoder batch 动态: 8GB 上 B*2560 显存受限, max=32; B>32 由 MFRDecoderTRT 拆块
BMIN, BOPT, BMAX = 1, 32, 32
# decoder past_len 动态 (opt=256 贴合长公式 ~408 步)
PMIN, POPT, PMAX = 1, 256, 2560

TRTEXEC = None  # 延迟到 main() 定位，避免 import 期报错


# ---- decoder past shape 构造（复刻 build_decoder_engines.py） ----------------

def _past_shape(name, B, past_len):
    if name == "input_ids":
        return f"{B}x1"
    if name == "attention_mask":
        return f"{B}x{past_len + 1}"
    if name.endswith("self_k") or name.endswith("self_v"):
        return f"{B}x16x{past_len}x32"
    if name.endswith("cross_k") or name.endswith("cross_v"):
        return f"{B}x16x{CROSS}x32"
    raise ValueError(name)


_past_inputs = ["input_ids", "attention_mask"]
for _i in range(N_LAYERS):
    for _tag in ["self_k", "self_v", "cross_k", "cross_v"]:
        _past_inputs.append(f"past_{_i}_{_tag}")


def _build_past_shapes(B, past_len):
    return ",".join(f"{n}:{_past_shape(n, B, past_len)}" for n in _past_inputs)


# ---- 通用 trtexec 调用 -------------------------------------------------------

def _run(tag, onnx, engine, minS, optS, maxS, fp16=True, workspace=None):
    """调 trtexec 构建单个引擎；日志落 engines_bin/build_<tag>.log，只打印摘要。"""
    if not os.path.exists(onnx):
        print(f"[{tag}] ✗ ONNX 缺失，跳过: {onnx}（先跑对应 export_*.py）")
        return False
    args = [TRTEXEC, f"--onnx={onnx}", f"--saveEngine={engine}"]
    if fp16:
        args.append("--fp16")
    args += [f"--minShapes={minS}", f"--optShapes={optS}", f"--maxShapes={maxS}"]
    if workspace:
        args.append(f"--memPoolSize=workspace:{workspace}")

    log = os.path.join(str(_env.ENGINES_DIR), f"build_{tag}.log")
    print(f"\n[{tag}] building{' (fp16)' if fp16 else ' (tf32)'} -> {engine}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = open(log, encoding="utf-8", errors="replace").read()
    passed = "PASSED" in txt and p.returncode == 0
    gpu = [l.strip() for l in txt.splitlines() if "GPU Compute Time" in l]
    print(f"[{tag}] {'PASSED' if passed else 'FAILED (rc=%d)' % p.returncode}  {dt:.0f}s  log={log}")
    for l in gpu:
        print("   ", l)
    if not passed:
        for l in txt.splitlines()[-15:]:
            print("   |", l)
    return passed


# ---- 各引擎构建函数 ---------------------------------------------------------

def build_decoder_init():
    onnx = _env.onnx_path("decoder_init.onnx")
    eng = _env.engine_path("decoder_init_fp16.engine")
    mn = f"input_ids:{BMIN}x1,attention_mask:{BMIN}x1,encoder_hidden_states:{BMIN}x{CROSS}x512"
    op = f"input_ids:{BOPT}x1,attention_mask:{BOPT}x1,encoder_hidden_states:{BOPT}x{CROSS}x512"
    mx = f"input_ids:{BMAX}x1,attention_mask:{BMAX}x1,encoder_hidden_states:{BMAX}x{CROSS}x512"
    return _run("decoder_init", onnx, eng, mn, op, mx, fp16=True, workspace=2048)


def build_decoder_past():
    onnx = _env.onnx_path("decoder_with_past.onnx")
    eng = _env.engine_path("decoder_with_past_fp16.engine")
    return _run("decoder_with_past", onnx, eng,
                _build_past_shapes(BMIN, PMIN),
                _build_past_shapes(BOPT, POPT),
                _build_past_shapes(BMAX, PMAX),
                fp16=True, workspace=2048)


def build_encoder():
    onnx = _env.onnx_path("encoder_ppformulanet.onnx")
    eng = _env.engine_path("encoder_ppformulanet_fp16.engine")
    # encoder max_batch=16（config banner），opt=8；pixel_values [B,1,384,384]
    mn = "pixel_values:1x1x384x384"
    op = "pixel_values:8x1x384x384"
    mx = "pixel_values:16x1x384x384"
    return _run("encoder", onnx, eng, mn, op, mx, fp16=True)


def build_dbnet():
    onnx = _env.onnx_path("dbnet.onnx")
    eng = _env.engine_path("dbnet.engine")
    # fast_ops/README: --fp16, 检测框对齐不受 fp16 影响（bbox 逐像素一致）
    mn = "x:1x3x96x128"
    op = "x:8x3x384x672"
    mx = "x:16x3x736x960"
    return _run("dbnet", onnx, eng, mn, op, mx, fp16=True)


def build_crnn(precision="tf32"):
    onnx = _env.onnx_path("crnn.onnx")
    # fast_ops/README: tf32(默认,不加--fp16,逐字对齐CPU) / fp16(快,极小字符抖动)
    mn = "x:1x3x48x16"
    op = "x:8x3x48x320"
    mx = "x:16x3x48x2560"
    ok = True
    if precision in ("tf32", "both"):
        ok &= _run("crnn_tf32", onnx, _env.engine_path("crnn.engine"), mn, op, mx, fp16=False)
    if precision in ("fp16", "both"):
        ok &= _run("crnn_fp16", onnx, _env.engine_path("crnn_fp16.engine"), mn, op, mx, fp16=True)
    return ok


_BUILDERS = {
    "decoder_init": build_decoder_init,
    "decoder_with_past": build_decoder_past,
    "encoder": build_encoder,
    "dbnet": build_dbnet,
    # crnn 单独处理（有 precision 参数）
}


def main():
    global TRTEXEC
    p = argparse.ArgumentParser(description="Build all TensorRT engines for fast_mineru")
    p.add_argument("--only", default=None,
                   help="逗号分隔的引擎子集: decoder_init,decoder_with_past,encoder,dbnet,crnn")
    p.add_argument("--crnn", default="tf32", choices=["tf32", "fp16", "both"],
                   help="crnn 引擎精度 (默认 tf32，config 默认用 tf32 的 crnn.engine)")
    args = p.parse_args()

    TRTEXEC = _env.find_trtexec()
    print("trtexec =", TRTEXEC)
    print("engines_bin =", _env.ensure_engines_dir())

    targets = (args.only.split(",") if args.only
               else ["decoder_init", "decoder_with_past", "encoder", "dbnet", "crnn"])
    results = {}
    for t in targets:
        t = t.strip()
        if t == "crnn":
            results["crnn"] = build_crnn(args.crnn)
        elif t in _BUILDERS:
            results[t] = _BUILDERS[t]()
        else:
            print(f"[warn] 未知引擎: {t}（可选: {', '.join(list(_BUILDERS)+['crnn'])}）")

    print("\n=== 构建结果 ===")
    for t, ok in results.items():
        print(f"  {t:20s}: {'OK' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
