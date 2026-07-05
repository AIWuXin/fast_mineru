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
    """把 pipeline_analyze.BatchAnalyze 重绑为 cls（FastBatchAnalyze），返回原类以便还原。"""
    import mineru.backend.pipeline.pipeline_analyze as _pa
    orig = getattr(_pa, "BatchAnalyze", None)
    _pa.BatchAnalyze = cls
    return orig


def restore_batch_analyze(orig):
    """把 pipeline_analyze.BatchAnalyze 还原为 orig（install 返回值）。"""
    if orig is None:
        return
    import mineru.backend.pipeline.pipeline_analyze as _pa
    _pa.BatchAnalyze = orig


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
