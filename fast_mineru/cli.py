"""fast-mineru CLI —— 位置参数传单个 PDF 或含 PDF 的文件夹，自动多文档处理。

    fast-mineru <pdf_或_文件夹> [--output DIR] [--no-mfr-dec-trt] [--debug] [--bench]

流程：Pipeline(config) 构造一次(加载模型/引擎/预分配/warmup) → 每个 PDF 调 process()
→ rich 进度 + 每文档 stage 表 + process() 总耗时。区分 init 耗时(一次) 与 process 耗时。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config import PipelineConfig
from .console import console, rule, timing_table
from .pipeline import FastMineruPipeline


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
    p.add_argument("--no-whole-page-gpu", action="store_true",
                   help="关闭整页 GPU 常驻(FastBatchAnalyze)，回退 mineru 原生 OCR 编排(A/B 对比用)")
    return p


def main(argv=None) -> int:
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
        use_whole_page_gpu=not args.no_whole_page_gpu,
        debug=args.debug,
        output_dir=Path(args.output),
    )
    if args.engine_dir:
        cfg.engine_dir = Path(args.engine_dir)

    console.print(f"[cyan]发现 {len(pdfs)} 个 PDF，输出 → {args.output}")

    pipe = FastMineruPipeline(cfg)

    rule("推理")
    results = []
    total_process_ms = 0.0
    total_pages = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("处理文档", total=len(pdfs))
        for pdf in pdfs:
            prog.update(task, description=f"[cyan]{pdf.name}")
            r = pipe.process(pdf)
            results.append(r)
            total_process_ms += r["process_ms"]
            total_pages += r["pages"]
            prog.advance(task)
            if args.bench and r.get("stage_rows"):
                console.print(timing_table(
                    f"{_ellipsis(r['name'], 36)}  ({r['pages']}p, process {r['process_ms']/1000:.2f}s)",
                    r["stage_rows"], r["process_ms"]))
                dh = lambda e: f"hit={getattr(e,'hit','-')} miss={getattr(e,'miss','-')}" if e else "-"
                console.print(f"  [dim]TRT 命中: DBNet {dh(pipe._dbnet)} | CRNN {dh(pipe._crnn)}[/dim]")

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
