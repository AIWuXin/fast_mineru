# -*- coding: utf-8 -*-
"""诚实的 A/B 基准：baseline(全关加速) vs accel(全开)，同一 PDF、同一 warmup。

- 每个 mode 在**独立子进程**跑(避免 8GB 上两套模型 OOM，且各自干净显存)。
- 每个 mode 内 warmup 一次后**复用同一 FastMineruPipeline 对象连跑 N 次**取 min/均值
  (这正是 process() 零加载的意义：稳定、可比)。
- 跑完逐字比对两 mode 的 markdown，证明"不掉精度"。

用法：
    uv run python tools/ab_benchmark.py <pdf> [--repeat 3]
子进程模式(内部用)：
    uv run python tools/ab_benchmark.py --worker <mode> <pdf> <repeat> <outdir>
"""
import io
import sys
import os
import subprocess
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _worker(mode: str, pdf: str, repeat: int, outdir: str):
    """在子进程内：构造一次 pipeline，连跑 repeat 次，输出 JSON 计时到 stdout 末行。"""
    os.environ.setdefault("PYTHONUTF8", "1")
    from pathlib import Path
    from fast_mineru import FastMineruPipeline, PipelineConfig

    accel = (mode == "accel")
    cfg = PipelineConfig(
        use_mfr_decoder_trt=accel,
        use_dbnet_trt=accel,
        use_crnn_trt=accel,
        use_fast_ops=accel,
        warmup_pages=2,
        output_dir=Path(outdir),
    )
    pipe = FastMineruPipeline(cfg)
    times = []
    md_path = None
    for i in range(repeat):
        r = pipe.process(pdf)
        times.append(r["process_ms"])
        mds = list(Path(r["output_dir"]).glob("*.md"))
        if mds:
            md_path = str(mds[0])
        print(f"[{mode}] run {i+1}/{repeat}: {r['process_ms']/1000:.2f}s", flush=True)
    result = {
        "mode": mode,
        "init_ms": pipe._init_elapsed * 1000,
        "times_ms": times,
        "min_ms": min(times),
        "avg_ms": sum(times) / len(times),
        "pages": r["pages"],
        "md_path": md_path,
        "injected": {"mfr": pipe._injected, "dbnet": pipe._dbnet_injected, "crnn": pipe._crnn_injected},
    }
    print("###RESULT###" + json.dumps(result), flush=True)


def _run_mode(mode: str, pdf: str, repeat: int) -> dict:
    outdir = f"ab_out_{mode}"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", __file__, "--worker", mode, pdf, str(repeat), outdir],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("[") or "run" in line:
            print("  " + line)
        if line.startswith("###RESULT###"):
            return json.loads(line[len("###RESULT###"):])
    print(f"  [!] {mode} 无结果，stderr 末尾：")
    print("\n".join("    " + l for l in proc.stderr.splitlines()[-15:]))
    return {}


def _diff_md(a: str, b: str):
    if not a or not b or not os.path.exists(a) or not os.path.exists(b):
        return None
    ta = open(a, encoding="utf-8").read()
    tb = open(b, encoding="utf-8").read()
    if ta == tb:
        return ("identical", 0)
    # 逐行差异计数
    la, lb = ta.splitlines(), tb.splitlines()
    diff = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
    return ("differ", diff)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
        return

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()

    print(f"=== A/B 基准: {os.path.basename(args.pdf)}  (每 mode warmup 一次 + 连跑 {args.repeat} 次) ===\n")

    print(">>> baseline (全关加速, 纯 torch)")
    base = _run_mode("baseline", args.pdf, args.repeat)
    print("\n>>> accel (MFR-dec + DBNet + CRNN TRT 全开)")
    acc = _run_mode("accel", args.pdf, args.repeat)

    if not base or not acc:
        print("\n[!] 某 mode 失败，无法比较")
        return

    print("\n" + "=" * 60)
    print(f"{'指标':<20}{'baseline':>15}{'accel':>15}")
    print("-" * 60)
    print(f"{'init 一次 (s)':<20}{base['init_ms']/1000:>15.1f}{acc['init_ms']/1000:>15.1f}")
    print(f"{'process min (s)':<20}{base['min_ms']/1000:>15.2f}{acc['min_ms']/1000:>15.2f}")
    print(f"{'process avg (s)':<20}{base['avg_ms']/1000:>15.2f}{acc['avg_ms']/1000:>15.2f}")
    spd = base["min_ms"] / acc["min_ms"] if acc["min_ms"] else 0
    saved = (base["min_ms"] - acc["min_ms"]) / 1000
    print("-" * 60)
    print(f"加速比 (min): {spd:.2f}x   |   每篇省 {saved:.2f}s   |   页数 {base['pages']}")
    print(f"accel 注入: {acc['injected']}")

    dr = _diff_md(base.get("md_path"), acc.get("md_path"))
    print("-" * 60)
    if dr is None:
        print("精度对比: 无法比对(缺 md)")
    elif dr[0] == "identical":
        print("精度对比: ✓ markdown 逐字完全一致(不掉精度)")
    else:
        print(f"精度对比: markdown 有 {dr[1]} 行不同(需人工看是否公式差异)")
    print("=" * 60)


if __name__ == "__main__":
    main()
