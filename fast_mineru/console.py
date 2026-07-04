"""rich Console 单例 + 统一的表格 / 面板 / 计时上下文。

全项目统一 `from fast_mineru.console import console`，别到处 print。
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager

# Windows 下 stdout 默认 GBK，rich 输出 ✓/─ 等字符会 UnicodeEncodeError。
# 重配为 UTF-8(重定向到文件或管道时尤其必要)。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # py3.7+ TextIOWrapper
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# legacy_windows=False：强制走 ANSI 渲染路径而非 GBK 的 win32 legacy renderer。
console = Console(legacy_windows=False)


def rule(title: str):
    console.rule(f"[bold cyan]{title}")


def panel(body: str, title: str = "", style: str = "cyan"):
    console.print(Panel(body, title=title, border_style=style, box=box.ROUNDED))


def kv_panel(title: str, items: dict, style: str = "cyan"):
    """键值面板：设备/显存/引擎/config 等启动信息。"""
    body = "\n".join(f"[bold]{k:<16}[/bold] {v}" for k, v in items.items())
    panel(body, title=title, style=style)


def timing_table(title: str, rows: list[tuple], total_ms: float | None = None) -> Table:
    """计时表：rows = [(stage, calls, wall_ms, pct), ...]。替代手绘 ASCII 表。"""
    t = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold cyan")
    t.add_column("stage", style="white")
    t.add_column("calls", justify="right", style="dim")
    t.add_column("wall (ms)", justify="right", style="green")
    t.add_column("%", justify="right", style="yellow")
    for stage, calls, wall_ms, pct in rows:
        t.add_row(str(stage), str(calls), f"{wall_ms:.1f}", f"{pct:.1f}")
    if total_ms is not None:
        t.add_row("[bold]total[/bold]", "", f"[bold]{total_ms:.1f}[/bold]", "100.0")
    return t


class Timer:
    """累计计时器：with timer.section('2_mfr'): ... 自动汇总。"""

    def __init__(self):
        self.acc: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000
            self.acc[name] = self.acc.get(name, 0.0) + dt
            self.calls[name] = self.calls.get(name, 0) + 1

    def add(self, name: str, ms: float):
        self.acc[name] = self.acc.get(name, 0.0) + ms
        self.calls[name] = self.calls.get(name, 0) + 1

    def total(self) -> float:
        return sum(self.acc.values())

    def rows(self) -> list[tuple]:
        tot = self.total() or 1.0
        return [
            (name, self.calls.get(name, 1), ms, ms / tot * 100)
            for name, ms in sorted(self.acc.items(), key=lambda x: -x[1])
        ]

    def reset(self):
        self.acc.clear()
        self.calls.clear()
