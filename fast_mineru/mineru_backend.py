"""mineru_backend —— fast_mineru 与 MinerU 之间的**唯一接缝**。

fast_mineru 的真正推理计算（engines/）已完全自持；对 MinerU 的耦合只剩两类：
1. **符号依赖**：BatchAnalyze 基类 / 编排入口 / 一堆前后处理 helper / 输出 IO / 枚举常量。
2. **反射进单例**：从 AtomModelSingleton / ModelSingleton 里捞出模型实例做显式注入。

本模块把这两类全部收敛到一处：

- **Part 1（懒再导出）**：`_LAZY` 注册表列出 fast_mineru 用到的**每一个** mineru 符号，
  配合 PEP 562 模块级 `__getattr__` 做**首次访问才导入**的再导出。调用方一律
  `from .mineru_backend import X`，看不到任何 `mineru.*` 内部路径；MinerU 升级只需改这一处。

- **Part 2（句柄 helper）**：把"reach into singleton 取模型 / 重绑 BatchAnalyze / 统计页数"
  等原本散落在 pipeline.py、_stagetimer.py 的丑逻辑上提为干净函数，内部才碰 mineru。

**关键不变量**：本模块**绝不在 import 时触碰 mineru**——`_LAZY` 是纯数据、helper 内部懒加载。
这样 `import fast_mineru` 不会提前触发 mineru 模型创建，保证 `PipelineConfig.apply_env()`
（设置 MINERU_FORMULA_CH_SUPPORT 等门控）先于任何 mineru 导入执行，行为与重构前逐字节一致。
"""
from __future__ import annotations

import importlib
from concurrent.futures.process import BrokenProcessPool

# ── Part 1：懒再导出注册表 ─────────────────────────────────────────────────
# name → (module_path, attr_name)。以 `grep -rhoE "from mineru\S* import"` 全量清单登记，不遗漏。
_LAZY: dict[str, tuple[str, str]] = {
    # 编排入口 / 单例
    "AtomModelSingleton":            ("mineru.backend.pipeline.model_init", "AtomModelSingleton"),
    "run_layout_inference":          ("mineru.backend.pipeline.model_init", "run_layout_inference"),
    "run_mfr_inference":             ("mineru.backend.pipeline.model_init", "run_mfr_inference"),
    "run_ocr_inference":             ("mineru.backend.pipeline.model_init", "run_ocr_inference"),
    "AtomicModel":                   ("mineru.backend.pipeline.model_list", "AtomicModel"),
    "ModelSingleton":                ("mineru.backend.pipeline.pipeline_analyze", "ModelSingleton"),
    "batch_image_analyze":           ("mineru.backend.pipeline.pipeline_analyze", "batch_image_analyze"),
    "doc_analyze_streaming":         ("mineru.backend.pipeline.pipeline_analyze", "doc_analyze_streaming"),
    # BatchAnalyze 基类 + 批大小常量
    "BatchAnalyze":                  ("mineru.backend.pipeline.batch_analyze", "BatchAnalyze"),
    "LAYOUT_BASE_BATCH_SIZE":        ("mineru.backend.pipeline.batch_analyze", "LAYOUT_BASE_BATCH_SIZE"),
    "MFR_BASE_BATCH_SIZE":           ("mineru.backend.pipeline.batch_analyze", "MFR_BASE_BATCH_SIZE"),
    "OCR_DET_BASE_BATCH_SIZE":       ("mineru.backend.pipeline.batch_analyze", "OCR_DET_BASE_BATCH_SIZE"),
    "TABLE_Wired_Wireless_CLS_BATCH_SIZE": ("mineru.backend.pipeline.batch_analyze",
                                            "TABLE_Wired_Wireless_CLS_BATCH_SIZE"),
    # bbox / model / ocr 前后处理 helper
    "normalize_to_int_bbox":         ("mineru.utils.bbox_utils", "normalize_to_int_bbox"),
    "_get_int_bbox":                 ("mineru.utils.model_utils", "_get_int_bbox"),
    "clean_vram":                    ("mineru.utils.model_utils", "clean_vram"),
    "get_res_list_from_layout_res":  ("mineru.utils.model_utils", "get_res_list_from_layout_res"),
    "crop_img":                      ("mineru.utils.model_utils", "crop_img"),
    "OcrConfidence":                 ("mineru.utils.ocr_utils", "OcrConfidence"),
    "TEXT_REC_ROTATE_RATIO":         ("mineru.utils.ocr_utils", "TEXT_REC_ROTATE_RATIO"),
    "calculate_is_angle":            ("mineru.utils.ocr_utils", "calculate_is_angle"),
    "get_adjusted_mfdetrec_res":     ("mineru.utils.ocr_utils", "get_adjusted_mfdetrec_res"),
    "merge_det_boxes":               ("mineru.utils.ocr_utils", "merge_det_boxes"),
    "sorted_boxes":                  ("mineru.utils.ocr_utils", "sorted_boxes"),
    "update_det_boxes":              ("mineru.utils.ocr_utils", "update_det_boxes"),
    "get_ocr_result_list":          ("mineru.utils.ocr_utils", "get_ocr_result_list"),
    "get_crop_np_img":               ("mineru.utils.pdf_image_tools", "get_crop_np_img"),
    # MFR 内部
    "build_mfr_batch_groups":        ("mineru.model.mfr.pp_formulanet_plus_m.predict_formula",
                                      "build_mfr_batch_groups"),
    "DonutSwinModelOutput":          ("mineru.model.utils.pytorchocr.modeling.backbones.rec_donut_swin",
                                      "DonutSwinModelOutput"),
    # 输出 IO / 枚举
    "FileBasedDataWriter":           ("mineru.data.data_reader_writer", "FileBasedDataWriter"),
    "MakeMode":                      ("mineru.utils.enum_class", "MakeMode"),
    "prepare_env":                   ("mineru.cli.common", "prepare_env"),
    "_process_output":               ("mineru.cli.common", "_process_output"),
    # PDF 渲染 / pdfium
    "shutdown_pdf_render_executor":  ("mineru.utils.pdf_image_tools", "shutdown_pdf_render_executor"),
    "open_pdfium_document":          ("mineru.utils.pdfium_guard", "open_pdfium_document"),
    "get_pdfium_document_page_count": ("mineru.utils.pdfium_guard", "get_pdfium_document_page_count"),
    "close_pdfium_document":         ("mineru.utils.pdfium_guard", "close_pdfium_document"),
}


def __getattr__(name: str):
    """PEP 562：首次访问 mineru 符号时才 import，并缓存进本模块 globals。"""
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = spec
    obj = getattr(importlib.import_module(module_path), attr)
    globals()[name] = obj  # 缓存，后续走常规属性查找
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


# ── Part 2：模型句柄 / 编排 helper（内部懒加载，行为与重构前逐字节一致）───────────

def get_mfr_head(device: str):
    """定位 pp_formulanet 的 head(PPFormulaNet_Head)。unimernet 无 generate_export 返回 None。"""
    try:
        from mineru.backend.pipeline.model_init import AtomModelSingleton
        from mineru.backend.pipeline.model_list import AtomicModel
        mfr = AtomModelSingleton().get_atom_model(
            atom_model_name=AtomicModel.MFR, device=device)
    except Exception:
        return None
    net = getattr(mfr, "net", None)
    head = getattr(net, "head", None) if net is not None else None
    if head is None or not hasattr(head, "generate_export"):
        return None
    return head


def get_mfr_backbone(device: str):
    """定位 pp_formulanet 的 backbone(encoder) + DonutSwinModelOutput。返回 (backbone, cls) 或 (None,None)。"""
    try:
        from mineru.backend.pipeline.model_init import AtomModelSingleton
        from mineru.backend.pipeline.model_list import AtomicModel
        mfr = AtomModelSingleton().get_atom_model(
            atom_model_name=AtomicModel.MFR, device=device)
    except Exception:
        return None, None
    net = getattr(mfr, "net", None)
    backbone = getattr(net, "backbone", None) if net is not None else None
    if backbone is None:
        return None, None
    try:
        from mineru.model.utils.pytorchocr.modeling.backbones.rec_donut_swin import (
            DonutSwinModelOutput,
        )
    except Exception:
        return None, None
    return backbone, DonutSwinModelOutput


def get_mfr_model(device: str):
    """取 MFR atom model 实例（供 MFR 预处理 GPU 注入）。失败抛出，交由调用方兜底。"""
    from mineru.backend.pipeline.model_init import AtomModelSingleton
    from mineru.backend.pipeline.model_list import AtomicModel
    return AtomModelSingleton().get_atom_model(
        atom_model_name=AtomicModel.MFR, device=device)


def iter_ocr_models() -> list:
    """遍历所有已创建的 OCR atom model 实例。

    OCR 的 singleton key 含 det_db_box_thresh/unclip_ratio/merge/lang，MinerU 会用不同参数
    建多个 OCR 实例（主路径、表格路径等），各有独立 net。注入必须覆盖全部已建实例。
    """
    try:
        from mineru.backend.pipeline.model_init import AtomModelSingleton
        from mineru.backend.pipeline.model_list import AtomicModel
    except Exception:
        return []
    models = getattr(AtomModelSingleton(), "_models", {})
    out = []
    for key, model in list(models.items()):
        name = key[0] if isinstance(key, tuple) else key
        if name == AtomicModel.OCR:
            out.append(model)
    return out


def iter_atom_models() -> list:
    """返回 [(atom_name, model_obj), ...]（供 StageTimer 包裹各 atom model 计时）。"""
    try:
        from mineru.backend.pipeline.model_init import AtomModelSingleton
    except Exception:
        return []
    mgr = AtomModelSingleton()
    out = []
    for key, obj in list(getattr(mgr, "_models", {}).items()):
        name = key[0] if isinstance(key, tuple) else key
        out.append((name, obj))
    return out


def iter_pipeline_layout_mfr() -> list:
    """返回 MineruPipelineModel 直接引用的 (attr_name, submodel) —— layout_model / mfr_model。

    这些不经 get_atom_model，需从 ModelSingleton 里取（供 StageTimer 补计时）。
    """
    try:
        from mineru.backend.pipeline.pipeline_analyze import ModelSingleton
    except Exception:
        return []
    out = []
    try:
        for model in list(getattr(ModelSingleton(), "_models", {}).values()):
            for attr in ("layout_model", "mfr_model"):
                sub = getattr(model, attr, None)
                if sub is not None:
                    out.append((attr, sub))
    except Exception:
        pass
    return out


def warmup(images_with_extra_info: list, *, formula_enable: bool, table_enable: bool):
    """跑一批 warmup 页，一次性创建并预热全部 atom model（转发 batch_image_analyze）。"""
    from mineru.backend.pipeline.pipeline_analyze import batch_image_analyze
    return batch_image_analyze(
        images_with_extra_info,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )


def analyze_streaming(**kwargs):
    """转发 doc_analyze_streaming（PDF→分页→分析 的整条编排入口）。"""
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
    return doc_analyze_streaming(**kwargs)


def install_fast_batch_analyze(cls):
    """把 FastBatchAnalyze 装到 mineru 的 BatchAnalyze 使用点，返回原类以便还原。

    MinerU 3.x 的 batch_image_analyze / doc_analyze_streaming 在**函数体内**做
    `from .batch_analyze import BatchAnalyze` 局部导入，因此重绑
    pipeline_analyze.BatchAnalyze 模块属性是无效的（该属性根本不存在）。
    必须重绑 `mineru.backend.pipeline.batch_analyze` 模块上的 BatchAnalyze 属性，
    局部导入才会拿到 cls。cls 是原类的子类，模块属性替换不影响已创建的类对象。
    """
    import mineru.backend.pipeline.batch_analyze as _ba
    orig = getattr(_ba, "BatchAnalyze", None)
    _ba.BatchAnalyze = cls
    return orig


def restore_batch_analyze(orig):
    """把 batch_analyze.BatchAnalyze 还原为 orig（install 返回值）。"""
    if orig is None:
        return
    import mineru.backend.pipeline.batch_analyze as _ba
    _ba.BatchAnalyze = orig


def shutdown_render_executor():
    """关闭 PDF 渲染 worker 进程池（转发 shutdown_pdf_render_executor）。"""
    from mineru.utils.pdf_image_tools import shutdown_pdf_render_executor
    shutdown_pdf_render_executor()


def count_pages(pdf_bytes: bytes) -> int:
    """统计 PDF 页数（pdfium_guard 三件套）。失败返回 0。"""
    try:
        import pypdfium2 as pdfium
        from mineru.utils.pdfium_guard import (
            open_pdfium_document, get_pdfium_document_page_count, close_pdfium_document)
        doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
        n = get_pdfium_document_page_count(doc)
        close_pdfium_document(doc)
        return n
    except Exception:
        return 0


# ── Part 3：路径版 pdfium 加载 + 输出进程池 worker（合批加速，2026-07-21）─────
#
# 动机（实测 10×112MB PDF 合批，"其它(渲染/IO/后处理)"占 66% 墙钟）：
# 1. mineru 的 _load_images_from_pdf_bytes_range 把 pdf_bytes **按任务** pickle 进进程池：
#    每篇文档 3 个 range 任务 × 112MB，10 篇合批 ≈ 3.4GB IPC。源文件就在磁盘上，
#    传路径让 worker 自己打开即可归零这部分 IPC。
# 2. MAX_PDF_RENDER_PROCESSES 硬编码 3，20 核机器上光栅化严重吃不饱。
# 3. _process_output(pypdf×2 + 原PDF落盘 + md) 是纯 Python CPU 活，线程池被 GIL 卡死，
#    换进程池才能真正与 GPU 推理重叠；pdf_bytes 同样从路径读，只 pickle middle_json。
#
# 本模块顶层不碰 mineru(PEP 562)，这些函数被 spawn worker import 时也是轻量的。

def _load_images_from_pdf_path_worker(pdf_path, dpi, start_page_id, end_page_id, image_type):
    """渲染进程池 worker(必须模块级,可 pickle)：从磁盘路径打开 pdfium。"""
    from mineru.utils.pdf_image_tools import load_images_from_pdf_core
    return load_images_from_pdf_core(pdf_path, dpi, start_page_id, end_page_id, image_type)


def _render_and_chars_worker(pdf_path, dpi, start_page_id, end_page_id, image_type):
    """渲染+字符提取组合 worker(模块级,可 pickle)：光栅化之外顺带把本 range 每页的
    文本层 chars(get_page_chars,纯页面函数,与 layout 无关)一并提取。

    收益:txt_spans_extract 的 get_page_chars 实测 ~120ms/页(逐 char FFI + 两遍去重),
    原本在 append 线程(GIL 内)串行执行;挪到渲染 worker 后随预取流水真并行。
    chars 失败返回 None → 主线程回退现算(宁可慢,不能错)。
    """
    from mineru.utils.pdf_image_tools import load_images_from_pdf_core
    images = load_images_from_pdf_core(pdf_path, dpi, start_page_id, end_page_id, image_type)
    chars_list = None
    try:
        import pypdfium2 as pdfium
        from mineru.utils.pdfium_guard import (
            open_pdfium_document, close_pdfium_document, close_pdfium_child, pdfium_guard)
        from mineru.utils.pdf_text_tool import get_page_chars
        doc = open_pdfium_document(pdfium.PdfDocument, pdf_path)
        try:
            chars_list = []
            for i in range(start_page_id, end_page_id + 1):
                page = None
                try:
                    with pdfium_guard():
                        page = doc[i]
                    # 与 txt_spans_extract 内部调用同参(默认 quote_loosebox=True),
                    # textpage/count 由 get_page_chars 自开自关,产出逐字节一致。
                    pc = get_page_chars(page)
                    # 压缩为纯基础类型元组再 pickle:Bbox/自定义类对象的大量小对象
                    # pickle 极慢(实测 +150ms/页),元组几乎免费;主进程查缓存时还原。
                    pc["chars"] = [
                        (tuple(c["bbox"].bbox), c["char"], c["rotation"],
                         (c["font"]["name"], c["font"]["flags"],
                          c["font"]["size"], c["font"]["weight"]),
                         c["char_idx"])
                        for c in pc["chars"]
                    ]
                    chars_list.append(pc)
                finally:
                    if page is not None:
                        close_pdfium_child(page)
        finally:
            close_pdfium_document(doc)
    except Exception:
        chars_list = None
    return images, chars_list


def _load_images_from_pdf_path_range(pdf_path, dpi, start_page_id, end_page_id,
                                     image_type, timeout, threads):
    """_load_images_from_pdf_bytes_range 的路径版：任务载荷 bytes→路径，其余逻辑复刻。"""
    import mineru.utils.pdf_image_tools as pit
    from concurrent.futures import wait, ALL_COMPLETED
    from concurrent.futures.process import BrokenProcessPool
    from mineru.utils.os_env_config import get_load_images_threads, get_load_images_timeout

    if end_page_id < start_page_id:
        return []
    if timeout is None:
        timeout = get_load_images_timeout()
    if threads is None:
        threads = get_load_images_threads()

    actual_threads, page_ranges = pit._get_render_process_plan(start_page_id, end_page_id, threads)
    executor = pit._get_pdf_render_executor()
    recycle_executor = False
    collected_image_lists = []
    try:
        futures, future_to_range = [], {}
        for range_start, range_end in page_ranges:
            future = pit._submit_pdf_render_task(
                executor, _load_images_from_pdf_path_worker,
                pdf_path, dpi, range_start, range_end, image_type,
            )
            futures.append(future)
            future_to_range[future] = range_start

        _, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)
        if not_done:
            recycle_executor = True
            raise TimeoutError(
                f"PDF image rendering timeout after {timeout}s "
                f"for pages {start_page_id + 1}-{end_page_id + 1}"
            )
        all_results = [(future_to_range[f], f.result()) for f in futures]
        all_results.sort(key=lambda x: x[0])
        images_list = []
        for _, imgs in all_results:
            collected_image_lists.append(imgs)
            images_list.extend(imgs)
        collected_image_lists.clear()
        return images_list
    except BrokenProcessPool:
        recycle_executor = True
        raise
    except Exception:
        for images_list in collected_image_lists:
            pit._close_image_dicts(images_list)
        raise
    finally:
        if recycle_executor:
            pit._recycle_pdf_render_executor(executor, terminate_processes=True)


def _load_images_and_chars_range(pdf_path, dpi, start_page_id, end_page_id,
                                 image_type, timeout, threads):
    """_load_images_from_pdf_path_range 的"渲染+字符"版:返回 (images_list, chars_list)。
    chars_list 与 images_list 页序对齐;某 range 提取失败则该 range 的 chars 为 None。"""
    import mineru.utils.pdf_image_tools as pit
    from concurrent.futures import wait, ALL_COMPLETED
    from concurrent.futures.process import BrokenProcessPool
    from mineru.utils.os_env_config import get_load_images_threads, get_load_images_timeout

    if end_page_id < start_page_id:
        return [], None
    if timeout is None:
        timeout = get_load_images_timeout()
    if threads is None:
        threads = get_load_images_threads()

    actual_threads, page_ranges = pit._get_render_process_plan(start_page_id, end_page_id, threads)
    executor = pit._get_pdf_render_executor()
    recycle_executor = False
    collected_image_lists = []
    try:
        futures, future_to_range = [], {}
        for range_start, range_end in page_ranges:
            future = pit._submit_pdf_render_task(
                executor, _render_and_chars_worker,
                pdf_path, dpi, range_start, range_end, image_type,
            )
            futures.append(future)
            future_to_range[future] = range_start

        _, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)
        if not_done:
            recycle_executor = True
            raise TimeoutError(
                f"PDF image rendering timeout after {timeout}s "
                f"for pages {start_page_id + 1}-{end_page_id + 1}"
            )
        all_results = [(future_to_range[f], f.result()) for f in futures]
        all_results.sort(key=lambda x: x[0])
        images_list, chars_list = [], []
        chars_ok = True
        for _, (imgs, chars) in all_results:
            collected_image_lists.append(imgs)
            images_list.extend(imgs)
            if chars is None:
                chars_ok = False  # 任一 range 失败 → 整体回退主线程现算
            elif chars_ok:
                chars_list.extend(chars)
        collected_image_lists.clear()
        if not chars_ok or len(chars_list) != len(images_list):
            chars_list = None
        return images_list, chars_list
    except BrokenProcessPool:
        recycle_executor = True
        raise
    except Exception:
        for images_list in collected_image_lists:
            pit._close_image_dicts(images_list)
        raise
    finally:
        if recycle_executor:
            pit._recycle_pdf_render_executor(executor, terminate_processes=True)


# id(pdf_bytes) → 源文件路径。doc_analyze_streaming 原样持有我们传入的 bytes 对象，
# id 在运行期间稳定；每次合批结束必须 unregister，防止 GC 后 id 复用指错文件。
_pdf_path_registry: dict[int, str] = {}
_orig_load_images_from_pdf_doc = None


def register_pdf_path(pdf_bytes: bytes, path: str) -> None:
    _pdf_path_registry[id(pdf_bytes)] = path


def unregister_pdf_path(pdf_bytes: bytes) -> None:
    _pdf_path_registry.pop(id(pdf_bytes), None)


def install_path_based_pdf_loader(max_processes: int = 0) -> None:
    """把 pipeline_analyze.load_images_from_pdf_doc 换成"注册过路径就走路径版"的包装。
    幂等。max_processes>0 时同时放开 mineru 渲染进程数硬上限(默认 3)。"""
    global _orig_load_images_from_pdf_doc
    import os as _os
    import mineru.backend.pipeline.pipeline_analyze as pa
    import mineru.utils.pdf_image_tools as pit

    if max_processes > 0:
        pit.MAX_PDF_RENDER_PROCESSES = max_processes
        _os.environ["MINERU_PDF_RENDER_THREADS"] = str(max_processes)

    if _orig_load_images_from_pdf_doc is None:
        _orig_load_images_from_pdf_doc = pit.load_images_from_pdf_doc

    def fast_load_images_from_pdf_doc(pdf_doc, *args, **kwargs):
        pdf_bytes = kwargs.get("pdf_bytes")
        path = _pdf_path_registry.get(id(pdf_bytes)) if pdf_bytes is not None else None
        if path is None:
            return _orig_load_images_from_pdf_doc(pdf_doc, *args, **kwargs)
        try:
            start_page_id = kwargs.get("start_page_id", 0)
            end_page_id = kwargs.get("end_page_id", None)
            from mineru.utils.pdfium_guard import get_pdfium_document_page_count
            from mineru.utils.pdf_image_tools import get_end_page_id
            page_num = get_pdfium_document_page_count(pdf_doc)
            return _load_images_from_pdf_path_range(
                path,
                dpi=kwargs.get("dpi", pit.DEFAULT_PDF_IMAGE_DPI),
                start_page_id=start_page_id,
                end_page_id=get_end_page_id(end_page_id, page_num),
                image_type=kwargs.get("image_type", pit.ImageType.PIL),
                timeout=kwargs.get("timeout", None),
                threads=kwargs.get("threads", None),
            )
        except BrokenProcessPool:
            raise  # 池断裂交给上层重试逻辑
        except Exception as _e:
            # 路径版任何意外都回退原版 bytes 路径(宁可慢，不能错)
            import sys as _sys
            print(f"[fast_mineru] 路径版渲染回退 bytes 路径: {type(_e).__name__}: {_e}",
                  file=_sys.stderr)
            return _orig_load_images_from_pdf_doc(pdf_doc, *args, **kwargs)

    pa.load_images_from_pdf_doc = fast_load_images_from_pdf_doc


def lower_process_priority():
    """把当前进程降为低优先级(Windows BELOW_NORMAL / POSIX nice+10)。

    必须模块级:作为 ProcessPoolExecutor initializer 被 pickle 到 worker。
    """
    import sys
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
        else:
            import os
            os.nice(10)
    except Exception:
        pass


def install_render_pool_low_priority() -> None:
    """渲染 worker 进程降为低优先级(Windows BELOW_NORMAL / POSIX nice+10)。

    预取流水线下 8 个 pdfium 渲染进程与主进程 GPU 预处理并发抢核,stage wall
    实测膨胀 ~30%。渲染只要求"比 analyze 快",优先级让贤不影响吞吐,主进程
    预处理优先拿核。幂等;纯调度调整,无数值影响。
    """
    import mineru.utils.pdf_image_tools as pit
    if getattr(pit._create_pdf_render_executor, "_fast_mineru_lowpri", False):
        return

    def create_with_low_priority(max_workers):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        kw = {"initializer": lower_process_priority}
        if pit.is_windows_environment():
            return ProcessPoolExecutor(max_workers=max_workers, **kw)
        if multiprocessing.get_start_method() != "spawn":
            return ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"), **kw)
        return ProcessPoolExecutor(max_workers=max_workers, **kw)

    create_with_low_priority._fast_mineru_lowpri = True
    pit._create_pdf_render_executor = create_with_low_priority


def install_conditional_clean_memory(threshold_gb: float = 7.0) -> None:
    """把 pipeline_analyze.clean_memory(每个窗口末 empty_cache)换成阈值版。

    原版每个 64 页窗口结束都 torch.cuda.empty_cache()：释放全部缓存块,下一窗口
    重新 cudaMalloc(churn + 同步)。流式合批下窗口峰值本就被 rec 宽度预算封顶,
    reserved 稳态远低于阈值时清缓存纯属浪费。仅当 reserved 超过阈值才真清。
    幂等;阈值 <=0 时保持原版行为。重绑模块全局,LOAD_GLOBAL 即刻生效。
    """
    if threshold_gb <= 0:
        return
    import torch
    import mineru.backend.pipeline.pipeline_analyze as pa
    if getattr(pa.clean_memory, "_fast_mineru_conditional", False):
        return
    orig = pa.clean_memory
    limit = threshold_gb * 1024**3

    def conditional_clean_memory(device="cuda"):
        try:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                if torch.cuda.memory_reserved() > limit:
                    orig(device)
                return
        except Exception:
            pass
        orig(device)

    conditional_clean_memory._fast_mineru_conditional = True
    pa.clean_memory = conditional_clean_memory


def process_output_task(pdf_path, pdf_name, local_md_dir, local_image_dir, render,
                        middle_json, model_output):
    """输出进程池 worker(必须模块级,可 pickle)：纯 CPU 落盘，与主进程 GPU 推理真并行。

    pdf 从路径读(免 112MB pickle)；middle_json/model_output 走 pickle(几 MB)。
    各篇目录/writer 独立，无共享状态，进程间无并发写冲突。

    注意：PEP 562 模块 __getattr__ 只在**属性访问**时触发，函数体内的裸全局名
    (LOAD_GLOBAL) 不经过它 —— 所以这里必须显式 import，不能裸写 FileBasedDataWriter。
    """
    from pathlib import Path as _Path
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode
    from mineru.cli.common import _process_output
    pdf_bytes = _Path(pdf_path).read_bytes()
    md_writer = FileBasedDataWriter(local_md_dir)
    _process_output(
        pdf_info=middle_json["pdf_info"], pdf_bytes=pdf_bytes, pdf_file_name=pdf_name,
        local_md_dir=local_md_dir, local_image_dir=local_image_dir, md_writer=md_writer,
        f_draw_layout_bbox=render, f_draw_span_bbox=render, f_dump_orig_pdf=render,
        f_dump_md=render, f_dump_content_list=True, f_dump_middle_json=True,
        f_dump_model_output=render, f_make_md_mode=MakeMode.MM_MD,
        middle_json=middle_json, model_output=model_output, process_mode="pipeline",
    )
