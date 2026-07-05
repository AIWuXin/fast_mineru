"""StageTimer —— 把 process() 的 wall time 细分到各模型 stage，回答"时间花在哪"。

移植自 D:/project/MinerU/benchmark_pipeline.py 的 Bench 拦截器，但更聚焦：只测各 atom model
的 batch_predict/predict/ocr 调用 wall time(CPU 侧 perf_counter，不强制 sync 不串行化 GPU)。
process() 的总 wall 减去这些 stage 之和 = "其它"(PDF 光栅化/分类/后处理/markdown dump 等
**不可加速**的开销)。

用途：解释 benchmark_pipeline 的 "5-6s"(只计模型推理 stage) vs fast_mineru process 的
wall(含全链) 的差异——差就差在"其它"。
"""
from __future__ import annotations

import time
from functools import wraps


_STAGE_MAP = {
    "layout": "1_layout",
    "mfr": "2_mfr",
    "table_ori_cls": "3_table",
    "table_cls": "3_table",
    "wireless_table": "3_table",
    "wired_table": "3_table",
}


class StageTimer:
    """安装一组方法包裹，累计各 stage wall_ms + calls。install()/uninstall() 幂等。"""

    def __init__(self, deep: bool = False):
        self.deep = deep   # 额外拆 det 预处理 / rec resize_norm 的 CPU 子阶段
        self.wall_ms: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._patched: list[tuple] = []   # (obj, attr, original)
        self._orig_get_atom = None

    def _record(self, stage: str, dt_ms: float):
        self.wall_ms[stage] = self.wall_ms.get(stage, 0.0) + dt_ms
        self.calls[stage] = self.calls.get(stage, 0) + 1

    def _wrap(self, fn, stage: str):
        rec = self._record
        @wraps(fn)
        def timed(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            rec(stage, (time.perf_counter() - t0) * 1000)
            return r
        timed._stagetimer = True
        return timed

    def _patch_method(self, obj, attr, stage):
        fn = getattr(obj, attr, None)
        if fn is None or getattr(fn, "_stagetimer", False):
            return
        setattr(obj, attr, self._wrap(fn, stage))
        self._patched.append((obj, attr, fn))

    def _wrap_atom(self, obj, aname):
        stage = _STAGE_MAP.get(str(aname))
        if str(aname) == "ocr":
            td = getattr(obj, "text_detector", None)
            if td is not None:
                self._patch_method(td, "batch_predict", "4_ocr_det")
                if self.deep:
                    self._patch_method(td, "_preprocess_det_image", "4a_det_preprocess(CPU)")
            if getattr(obj, "ocr", None) is not None:
                self._patch_method(obj, "ocr", "5_ocr_rec")
            if self.deep:
                rec = getattr(obj, "text_recognizer", None)
                if rec is not None:
                    self._patch_method(rec, "resize_norm_img", "5a_rec_resize(CPU)")
            return
        if stage is None:
            return
        self._patch_method(obj, "batch_predict", stage)
        self._patch_method(obj, "predict", stage)

    def install(self):
        """包裹当前已建的所有 atom model + MineruPipelineModel 的 layout/mfr 子模型。"""
        from . import mineru_backend as mb
        for aname, obj in mb.iter_atom_models():
            self._wrap_atom(obj, aname)
        # MineruPipelineModel 直接引用的 layout/mfr 子模型(不经 get_atom_model)
        _stage_by_attr = {"layout_model": "1_layout", "mfr_model": "2_mfr"}
        for attr, sub in mb.iter_pipeline_layout_mfr():
            stage = _stage_by_attr.get(attr)
            if stage is not None:
                self._patch_method(sub, "batch_predict", stage)
                self._patch_method(sub, "predict", stage)
        return self

    def uninstall(self):
        for obj, attr, fn in reversed(self._patched):
            try:
                setattr(obj, attr, fn)
            except Exception:
                pass
        self._patched.clear()

    def rows(self, total_wall_ms: float | None = None):
        """返回 [(stage, calls, wall_ms, pct)]，含 '其它'(total - sum stages)。pct 相对 total。"""
        rows = [(k, self.calls[k], v) for k, v in self.wall_ms.items()]
        rows.sort(key=lambda x: x[0])
        stage_sum = sum(v for _, _, v in rows)
        base = total_wall_ms if total_wall_ms else stage_sum
        out = [(k, c, v, (v / base * 100 if base else 0)) for k, c, v in rows]
        if total_wall_ms:
            other = max(0.0, total_wall_ms - stage_sum)
            out.append(("其它(渲染/IO/后处理)", 1, other, other / base * 100 if base else 0))
        return out
