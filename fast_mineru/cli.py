"""fast-mineru CLI —— 位置参数传单个 PDF 或含 PDF 的文件夹，自动多文档处理。

    fast-mineru <pdf_或_文件夹> [--output DIR] [--no-mfr-dec-trt] [--debug] [--bench]

流程：Pipeline(config) 构造一次(加载模型/引擎/预分配/warmup) → 每个 PDF 调 process()
→ rich 进度 + 每文档 stage 表 + process() 总耗时。区分 init 耗时(一次) 与 process 耗时。

**Windows spawn 安全**：本模块顶层只 import 轻量标准库。MinerU 的 pdfium 渲染进程池
在 Windows 上是 spawn 模式，每个 worker 都会重新 import __main__(即本文件)——若顶层
import torch/tensorrt/fast_mineru，每个 worker 白付数秒~十几秒 import(每文档重建池一次)，
debug 时还会被调试器 trace 放大数倍。重 import 全部收进 main()。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _disp_w(s: str) -> int:
    """终端显示宽度(CJK/全角=2，其余=1)。"""
    from unicodedata import east_asian_width
    return sum(2 if east_asian_width(c) in ("W", "F") else 1 for c in s)


def _ellipsis(s: str, width: int) -> str:
    """按**显示宽度**中部省略：保留头尾(区分文档的关键信息)，中间用 … 顶掉。

    CJK 字符占 2 列，故按显示宽度而非字符数预算，避免终端里撑爆表格。
    """
    if _disp_w(s) <= width:
        return s
    if width <= 1:
        return "…"
    budget = width - 1  # 去掉 … 占的 1 列
    head_budget = budget * 2 // 3
    tail_budget = budget - head_budget

    def _take(seq, limit):
        out, used = [], 0
        for c in seq:
            w = 2 if _disp_w(c) == 2 else 1
            if used + w > limit:
                break
            out.append(c)
            used += w
        return "".join(out), used

    head, _ = _take(s, head_budget)
    tail_rev, _ = _take(reversed(s), tail_budget)
    tail = tail_rev[::-1]
    return head + "…" + tail


def _collect_pdfs(target: Path) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.pdf"))
    return []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fast-mineru",
        description="MinerU 加速推理：单 PDF 或文件夹批处理，全程 GPU 的 MFR decoder TRT 加速")
    p.add_argument("target", help="单个 PDF 路径，或含多个 PDF 的文件夹(递归收集)")
    p.add_argument("--output", "-o", default="fast_mineru_out", help="输出目录(默认 fast_mineru_out)")
    p.add_argument("--method", "-m", choices=["auto", "ocr", "txt"], default="auto",
                   help="PDF 解析方式：auto=自动分类(默认,与原版一致)；ocr=强制全页 OCR；txt=强制文本层抽取")
    p.add_argument("--engine-dir", default=None, help="TRT 引擎目录(默认包内 engines_bin/)")
    p.add_argument("--no-mfr-dec-trt", action="store_true", help="关闭 MFR-decoder TRT(回退 torch)")
    p.add_argument("--mfr-precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--warmup", type=int, default=2, help="init 期 warmup 页数")
    p.add_argument("--no-render", action="store_true",
                   help="跳过 markdown/画框PDF 渲染(纯测推理速度)。只出 middle_json/content_list")
    p.add_argument("--verbose-mineru", action="store_true",
                   help="保留 MinerU 内部 tqdm/日志(默认静音，只留 fast_mineru 的 rich 输出)")
    p.add_argument("--debug", action="store_true", help="打印每组 TRT 解码计时")
    p.add_argument("--bench", action="store_true", help="打印每文档 stage 计时表")
    p.add_argument("--whole-page-gpu", action="store_true",
                   help="开启整页 GPU 常驻(FastBatchAnalyze)。实验性：实测比原生编排慢约四成，"
                        "显存尖峰已由 rec 宽度预算分批根治，无需用它压显存")
    p.add_argument("--no-mfr-enc-trt", action="store_true",
                   help="关闭 MFR-encoder TRT，encoder 回退 torch(A/B 对比用)")
    p.add_argument("--torch-rec", action="store_true",
                   help="OCR-rec 整段回退 mineru 原生 torch(跳过 CRNN TRT + rec GPU 预处理，"
                        "rec 走 CPU resize_norm)。det/layout/MFR 仍 TRT+GPU。用于隔离验证 rec 是否为显存锯齿来源")
    p.add_argument("--no-prefetch", action="store_true",
                   help="关闭渲染预取流水线(回退串行窗口循环,A/B 对比用)")
    p.add_argument("--no-overlap-append", action="store_true",
                   help="关闭逐页后处理与 analyze 重叠(A/B 对比用)")
    p.add_argument("--clean-cache-threshold", type=float, default=7.0,
                   help="窗口末 empty_cache 的 reserved 阈值(GB);0=恢复 mineru 原版每窗口全量清")
    p.add_argument("--output-workers", type=int, default=4,
                   help="输出进程池 worker 数(默认 4)")
    return p


def main(argv=None) -> int:
    # 重 import 收在 main() 内：spawn worker 重新 import 本模块时不会执行到这里。
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    from fast_mineru.config import PipelineConfig
    from fast_mineru.console import console, rule, timing_table
    from fast_mineru.pipeline import FastMineruPipeline

    args = build_parser().parse_args(argv)

    target = Path(args.target)
    pdfs = _collect_pdfs(target)
    if not pdfs:
        console.print(f"[red]未找到 PDF: {target}")
        return 1

    cfg = PipelineConfig(
        use_mfr_decoder_trt=not args.no_mfr_dec_trt,
        mfr_precision=args.mfr_precision,
        warmup_pages=args.warmup,
        no_render=args.no_render,
        quiet_mineru=not args.verbose_mineru,
        stage_timing=args.bench,      # --bench 时细分各模型 stage wall
        use_whole_page_gpu=args.whole_page_gpu,
        use_mfr_encoder_trt=not args.no_mfr_enc_trt,
        use_torch_rec=args.torch_rec,
        parse_method=args.method,
        prefetch_render=not args.no_prefetch,
        overlap_append=not args.no_overlap_append,
        clean_cache_threshold_gb=args.clean_cache_threshold,
        output_workers=args.output_workers,
        debug=args.debug,
        output_dir=Path(args.output),
    )
    if args.engine_dir:
        cfg.engine_dir = Path(args.engine_dir)

    console.print(f"[cyan]发现 {len(pdfs)} 个 PDF，输出 → {args.output}")

    pipe = FastMineruPipeline(cfg)

    rule("推理")
    results = []
    total_pages = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("处理文档", total=len(pdfs))

        def _done(name):
            prog.update(task, description=f"[cyan]{name}")
            prog.advance(task)

        if len(pdfs) > 1:
            # 多文档真合批：单次 analyze_streaming 共享处理窗口 + 输出线程池重叠。
            results = pipe.process_many(pdfs, on_doc_done=_done)
        else:
            prog.update(task, description=f"[cyan]{pdfs[0].name}")
            results = [pipe.process(pdfs[0])]
            prog.advance(task)

        for r in results:
            total_pages += r["pages"]
        if args.bench:
            for r in results:
                if r.get("stage_rows"):
                    total_ms = max(x["process_ms"] for x in results)
                    console.print(timing_table(
                        f"{_ellipsis(r['name'], 36)}  ({total_pages}p, process {total_ms/1000:.2f}s)"
                        if len(results) > 1 else
                        f"{_ellipsis(r['name'], 36)}  ({r['pages']}p, process {r['process_ms']/1000:.2f}s)",
                        r["stage_rows"], total_ms if len(results) > 1 else r["process_ms"]))
                    dh = lambda e: f"hit={getattr(e,'hit','-')} miss={getattr(e,'miss','-')}" if e else "-"
                    console.print(f"  [dim]TRT 命中: DBNet {dh(pipe._dbnet)} | CRNN {dh(pipe._crnn)}[/dim]")
                    break  # 合批只有一张总表(挂在首篇)

    # 合批时各篇 process_ms 是"完成于"的累计时间点，总耗时取最后一篇的完成时间。
    total_process_ms = max((r["process_ms"] for r in results), default=0.0)

    # ---- 汇总 ----
    rule("汇总")
    from rich.table import Table
    from rich import box
    t = Table(box=box.SIMPLE_HEAVY, title_style="bold cyan")
    # 文档名中部省略：保留结尾区分信息(如 "副本 (2)")。_ellipsis 宽度 == 列不再二次截断。
    NAME_W = 46
    t.add_column("文档", style="white", no_wrap=True)
    t.add_column("页数", justify="right", style="dim")
    t.add_column("process 耗时", justify="right", style="green")
    t.add_column("pages/s", justify="right", style="yellow")
    for r in results:
        pps = r["pages"] / (r["process_ms"] / 1000) if r["process_ms"] > 0 else 0
        t.add_row(_ellipsis(r["name"], NAME_W), str(r["pages"]),
                  f"{r['process_ms']/1000:.2f}s", f"{pps:.2f}")
    console.print(t)

    avg = total_process_ms / len(results) / 1000 if results else 0
    pps_all = total_pages / (total_process_ms / 1000) if total_process_ms > 0 else 0
    console.print(
        f"\n[bold green]{len(results)} 个文档[/bold green]  "
        f"init 一次 [cyan]{pipe._init_elapsed:.1f}s[/cyan]  |  "
        f"process 总耗时 [bold]{total_process_ms/1000:.2f}s[/bold]  "
        f"平均 [bold]{avg:.2f}s/doc[/bold]  {pps_all:.2f} pages/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
