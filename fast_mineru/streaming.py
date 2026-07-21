"""streaming —— fast_mineru 自己的窗口编排(替代 mineru doc_analyze_streaming 的串行循环)。

为什么抽成自己的实现而不是 patch:
要改的是**循环结构本身**。mineru 的窗口循环是 渲染(N) → analyze(N) → append(N) 严格
串行:主线程在 pdfium 渲染等待(实测 ~29%)和逐页后处理(实测 ~16%)上干等,GPU/CPU 互相
空转。猴子补丁够不到循环内部,因此按原循环的窗口切分逻辑**逐行复刻**一份,加入三路流水:

    render(N+1)  ∥  analyze(N)  ∥  append(N-1)

- **render 线程**: 提前一个窗口提交 pdfium 光栅化(进程池),analyze 期间渲染池不闲置;
- **append 线程**: 逐页 MagicModel/cut_image/middle_json 纯 Python 后处理,在 analyze 的
  CUDA 同步间隙(GIL 释放)中并行;单线程队列保证 append/finalize 顺序与原实现完全一致;
- **finalize**(含 post-OCR 的 GPU rec)在 append 线程内按序执行;TRT 执行由
  engines.trt_base 的全局执行锁与主线程互斥,数值结果不变。

精度不变量:窗口切分、每页计算、append 顺序、finalize 相对文档完成点的时机,全部与原
实现一致 —— 输出逐字节相同。

线程安全要点:
- 主进程 pdfium(pdf_doc 取页/关闭)只被 append 线程触碰,且整段 append 持有
  _PDFIUM_LOCK;render 的正常路径(_load_images_from_pdf_path_range)完全不碰主进程
  pdfium;罕见的 bytes 回退路径也先拿这把锁。
- render/append 各一个单线程池,天然保序;append 积压最多一个窗口(提交前先收上一个
  的结果,顺带及时抛错),PIL 图像驻留内存有界。
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

# 主进程 pdfium 互斥:append 线程(取页/关文档) vs render 回退路径。
_PDFIUM_LOCK = threading.Lock()


def _lower_current_thread_priority():
    """append 工作线程降为 BELOW_NORMAL:它与主线程 analyze 的 Python 预处理争 GIL/核,
    优先级让贤可减少 stage wall 膨胀(append 只需在下一个 analyze 结束前跑完)。
    纯调度调整,无数值影响。"""
    import sys
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetThreadPriority(k.GetCurrentThread(), -1)  # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:
        pass


# ── 页面 chars 预取缓存 ─────────────────────────────────────────────────────
# txt_spans_extract 的 get_page_chars(~120ms/页,逐 char FFI+两遍去重)是纯页面函数,
# 由渲染 worker 随光栅化一并提取(_render_and_chars_worker),append 线程查缓存即用,
# 未命中(提取失败/旧路径)回退原版现算。append 单线程 → 模块级暂存无竞态。
_PAGE_CHARS_CACHE: dict = {}          # (id(pdf_doc), page_index) -> page_chars dict
_CURRENT_PAGE_CHARS = [None]          # 当前页预取结果(append 单线程)
_orig_get_page_chars = None
_orig_pmitpi = None


def _cache_aware_get_page_chars(page, textpage=None, quote_loosebox=True,
                                page_char_count=None):
    cached = _CURRENT_PAGE_CHARS[0]
    if cached is not None:
        return cached
    return _orig_get_page_chars(page, textpage=textpage, quote_loosebox=quote_loosebox,
                                page_char_count=page_char_count)


def _cache_aware_pmitpi(page_model_info, image_dict, page, image_writer, page_index,
                        ocr_enable=False):
    doc = getattr(page, "pdf", None)
    key = (id(doc), page_index) if doc is not None else None
    _CURRENT_PAGE_CHARS[0] = _PAGE_CHARS_CACHE.pop(key, None) if key else None
    try:
        return _orig_pmitpi(page_model_info, image_dict, page, image_writer,
                            page_index, ocr_enable=ocr_enable)
    finally:
        _CURRENT_PAGE_CHARS[0] = None


def _inflate_page_chars(pc):
    """把 worker 压缩的 page_chars(纯元组)还原为 get_page_chars 原版结构(Bbox 对象)。
    在预取线程执行(它大部分时间在等渲染 future,有 GIL 余量),append 线程零成本命中。"""
    from pdftext.schema import Bbox
    chars = [
        {
            "bbox": Bbox([b0, b1, b2, b3]),
            "char": ch,
            "rotation": rot,
            "font": {"name": fn, "flags": ff, "size": fs, "weight": fw},
            "char_idx": ci,
        }
        for (b0, b1, b2, b3), ch, rot, (fn, ff, fs, fw), ci in pc["chars"]
    ]
    return {
        "bbox": pc["bbox"], "width": pc["width"], "height": pc["height"],
        "rotation": pc["rotation"], "char_count": pc["char_count"], "chars": chars,
    }


def install_page_chars_cache() -> None:
    """安装 chars 预取缓存的两个薄包装(幂等)。命中时产出与原版逐字节一致
    (worker 用同一 get_page_chars 同参提取)。"""
    global _orig_get_page_chars, _orig_pmitpi
    if _orig_get_page_chars is not None:
        return
    import mineru.utils.span_pre_proc as sps
    import mineru.backend.pipeline.model_json_to_middle_json as m2m
    _orig_get_page_chars = sps.get_page_chars
    sps.get_page_chars = _cache_aware_get_page_chars
    _orig_pmitpi = m2m.page_model_info_to_page_info
    m2m.page_model_info_to_page_info = _cache_aware_pmitpi


def _plan_windows(doc_contexts, window_size: int):
    """窗口切分:与原循环逐 context 顺序取 min(剩余容量, 剩余页数) 完全一致。"""
    windows, cur, cap = [], [], window_size
    for ctx in doc_contexts:
        idx = 0
        while idx < ctx["page_count"]:
            if cap == 0:
                windows.append(cur)
                cur, cap = [], window_size
            take = min(cap, ctx["page_count"] - idx)
            cur.append((ctx, idx, take))
            idx += take
            cap -= take
    if cur:
        windows.append(cur)
    return windows


def analyze_streaming_fast(
    pdf_bytes_list,
    image_writer_list,
    lang_list,
    on_doc_ready,
    *,
    pdf_path_list=None,
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
    prefetch: bool = True,
    overlap_append: bool = True,
    low_priority: bool = False,
    prefetch_chars: bool = False,
):
    """doc_analyze_streaming 的流水线版。参数语义与原版一致;pdf_path_list 提供时
    渲染走路径版(worker 从磁盘打开,免 112MB/任务 pickle)。"""
    import pypdfium2 as pdfium
    import mineru.backend.pipeline.pipeline_analyze as pa
    import mineru.utils.pdf_image_tools as pit
    from mineru.backend.pipeline.model_json_to_middle_json import init_middle_json
    from mineru.utils.config_reader import get_processing_window_size
    from . import mineru_backend as mb

    if not (len(pdf_bytes_list) == len(image_writer_list) == len(lang_list)):
        raise ValueError("pdf_bytes_list, image_writer_list, and lang_list must have the same length")

    if prefetch_chars:
        install_page_chars_cache()  # 幂等;缓存包装仅在启用时安装

    # ---- contexts(复刻原实现) ----
    doc_contexts = []
    try:
        total_pages = 0
        for doc_index, (pdf_bytes, image_writer, lang) in enumerate(
            zip(pdf_bytes_list, image_writer_list, lang_list)
        ):
            ocr_enable = pa._get_ocr_enable(pdf_bytes, parse_method)
            pdf_doc = mb.open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
            try:
                page_count = mb.get_pdfium_document_page_count(pdf_doc)
                context = {
                    "doc_index": doc_index,
                    "pdf_bytes": pdf_bytes,
                    "pdf_doc": pdf_doc,
                    "page_count": page_count,
                    "pages_done": 0,
                    "middle_json": init_middle_json(),
                    "model_list": [],
                    "image_writer": image_writer,
                    "lang": lang,
                    "ocr_enable": ocr_enable,
                    "closed": False,
                }
            except Exception:
                mb.close_pdfium_document(pdf_doc)
                raise
            total_pages += page_count
            doc_contexts.append(context)

        if total_pages == 0:
            pa._emit_zero_page_contexts(doc_contexts, on_doc_ready)
            return

        window_size = get_processing_window_size(default=64)
        windows = _plan_windows(doc_contexts, window_size)
        pa._emit_zero_page_contexts(doc_contexts, on_doc_ready)

        # ---- render(预取线程):光栅化(+可选 chars 预取) ----
        def render_window(wi):
            batch_images, payloads = [], []
            for ctx, page_start, take in windows[wi]:
                page_end = page_start + take - 1
                path = pdf_path_list[ctx["doc_index"]] if pdf_path_list else None
                images_list = None
                if path is not None:
                    try:
                        if prefetch_chars:
                            images_list, chars_list = mb._load_images_and_chars_range(
                                path,
                                dpi=pit.DEFAULT_PDF_IMAGE_DPI,
                                start_page_id=page_start,
                                end_page_id=page_end,
                                image_type=pit.ImageType.PIL,
                                timeout=None,
                                threads=None,
                            )
                            if chars_list:
                                base = id(ctx["pdf_doc"])
                                for off, pc in enumerate(chars_list):
                                    _PAGE_CHARS_CACHE[(base, page_start + off)] = \
                                        _inflate_page_chars(pc)
                        else:
                            images_list = mb._load_images_from_pdf_path_range(
                                path,
                                dpi=pit.DEFAULT_PDF_IMAGE_DPI,
                                start_page_id=page_start,
                                end_page_id=page_end,
                                image_type=pit.ImageType.PIL,
                                timeout=None,
                                threads=None,
                            )
                    except BrokenProcessPool:
                        raise
                    except Exception:
                        images_list = None  # 回退原版(bytes/patched 包装)
                if images_list is None:
                    with _PDFIUM_LOCK:  # 原版要碰主进程 pdfium,与 append 线程互斥
                        images_list = pa.load_images_from_pdf_doc(
                            ctx["pdf_doc"],
                            start_page_id=page_start,
                            end_page_id=page_end,
                            image_type=pit.ImageType.PIL,
                            pdf_bytes=ctx["pdf_bytes"],
                        )
                for image_dict in images_list:
                    batch_images.append(
                        (image_dict["img_pil"], ctx["ocr_enable"], ctx["lang"]))
                payloads.append((ctx, images_list, page_start, take))
            return batch_images, payloads

        # ---- append(重叠线程):逐页后处理 + 文档完成即 finalize,严格保序 ----
        def append_window(payloads, batch_results):
            with _PDFIUM_LOCK:
                result_offset = 0
                for ctx, images_list, page_start, take in payloads:
                    result_slice = batch_results[result_offset: result_offset + take]
                    try:
                        pa.append_batch_results_to_middle_json(
                            ctx["middle_json"],
                            result_slice,
                            images_list,
                            ctx["pdf_doc"],
                            ctx["image_writer"],
                            page_start_index=page_start,
                            ocr_enable=ctx["ocr_enable"],
                            model_list=ctx["model_list"],
                            progress_bar=None,
                        )
                    finally:
                        pit._close_image_dicts(images_list)
                        images_list.clear()
                    result_offset += take
                    ctx["pages_done"] += take
                    if ctx["pages_done"] >= ctx["page_count"] and not ctx["closed"]:
                        pa._finalize_processing_window_context(ctx, on_doc_ready)

        render_ex = ThreadPoolExecutor(max_workers=1) if prefetch else None
        append_ex = None
        if overlap_append:
            _ex_kw = ({"initializer": _lower_current_thread_priority}
                      if low_priority else {})
            append_ex = ThreadPoolExecutor(max_workers=1, **_ex_kw)
        app_fut = None
        try:
            rend_fut = render_ex.submit(render_window, 0) if prefetch else None
            for wi in range(len(windows)):
                if prefetch:
                    try:
                        batch_images, payloads = rend_fut.result()
                    except BrokenProcessPool as e:
                        raise RuntimeError(f"BrokenProcessPool: {e}") from e
                    rend_fut = (render_ex.submit(render_window, wi + 1)
                                if wi + 1 < len(windows) else None)
                else:
                    batch_images, payloads = render_window(wi)

                batch_results = pa.batch_image_analyze(
                    batch_images,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )

                if append_ex is not None:
                    if app_fut is not None:
                        app_fut.result()  # 限积压 + 及时抛错(通常已并行跑完)
                    app_fut = append_ex.submit(append_window, payloads, batch_results)
                else:
                    append_window(payloads, batch_results)
            if app_fut is not None:
                app_fut.result()
        finally:
            if render_ex is not None:
                render_ex.shutdown(wait=True)
            if append_ex is not None:
                append_ex.shutdown(wait=True)
    finally:
        _PAGE_CHARS_CACHE.clear()
        for context in doc_contexts:
            if not context["closed"]:
                mb.close_pdfium_document(context["pdf_doc"])
                context["closed"] = True
