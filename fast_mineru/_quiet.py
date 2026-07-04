"""静音 MinerU 的第三方噪音，让 fast_mineru 的 rich 输出成为唯一进度层。

三类噪音：
1. **tqdm 进度条**(MinerU 内部 `tqdm(total=..., desc="Processing pages")` 等，无全局开关)
   → 全局 patch tqdm.__init__ 强制 disable=True(仅本进程，不改 MinerU 源码)。
2. **loguru DEBUG 日志**(模型路径/gc/分类失败等)→ 提升 loguru 级别到 WARNING/ERROR。
3. **pdfium C 库 fd 级 stderr**(渲染画框PDF时的 `Multiple definitions ... for key /Ascent`)
   → no_render 时自然消失；渲染时用 suppress_c_stderr() fd 重定向过滤。
"""
from __future__ import annotations

import os
import sys
import contextlib


def silence_mineru(level: str = "WARNING"):
    """一次性静音 MinerU 的 tqdm + loguru。幂等。"""
    _disable_tqdm()
    _silence_loguru()


def _disable_tqdm():
    """patch tqdm，让 MinerU 内部所有进度条静默(强制 disable=True)。"""
    try:
        import tqdm as _tqdm_mod
        from tqdm import tqdm as _tqdm_cls
    except Exception:
        return
    if getattr(_tqdm_cls, "_fast_mineru_silenced", False):
        return
    _orig_init = _tqdm_cls.__init__

    def _init(self, *args, **kwargs):
        kwargs["disable"] = True
        return _orig_init(self, *args, **kwargs)

    _tqdm_cls.__init__ = _init
    _tqdm_cls._fast_mineru_silenced = True
    # tqdm.auto / tqdm.std 也指向同一类，patch 一次即可覆盖多数导入路径。


def _silence_loguru():
    """彻底移除 loguru 的所有 handler，让 MinerU 的日志(含 ERROR)全部丢弃。

    关键：不能把 sink 留在 sys.stderr —— suppress_c_stderr() 会临时把 fd2 重定向到
    devnull，若此时 loguru 往 sys.stderr 写(且级别>阈值，如 pdf_classify 的 ERROR)，
    Windows 上会 WinError 1(函数不正确)并触发 handler 崩溃级联。直接 remove 全部 handler，
    loguru 无 sink → 无写入 → 与 fd 重定向彻底解耦。
    """
    try:
        from loguru import logger
    except Exception:
        return
    try:
        logger.remove()  # 移除全部 handler；MinerU 的 logger.xxx 变为 no-op
    except Exception:
        pass


@contextlib.contextmanager
def suppress_c_stderr():
    """fd 级抑制 C 库(pdfium/reportlab)直写 stderr 的噪音(/Ascent 等)。

    Python logging 抓不到这些——它们是 write(2) 到 fd 2。这里把 fd 2 临时重定向到 devnull。
    仅用于渲染阶段(no_render 时无需)。
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        # stderr 已被替换成非真实 fd(如 rich 捕获)——退化为不抑制。
        yield
        return
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull)
