"""FastMineruPipeline —— 把整条 MinerU 管线包装成一个简洁类。

设计契约(用户核心诉求)：
- **__init__ 完成一切前置**：加载 MinerU 全部 atom model + 反序列化 TRT 引擎 + 显式注入
  TRT decoder 到 MFR head(无 monkey-patch，一次绑定) + 预分配复用资源 + warmup。
- **process() 纯推理**：无模型加载 / 无引擎反序列化 / 无重复初始化。每篇 PDF 只做
  读取→前向→后处理，返回结构化结果 + 每 stage 计时 + process 总耗时。
- **process_many()**：多文档复用同一组模型与引擎，模型只加载一次。
"""
from __future__ import annotations

import gc
import time
import contextlib
from pathlib import Path

import torch

from . import mineru_backend as mb
from .config import PipelineConfig
from .console import console, kv_panel, timing_table, Timer, rule
from .engines import MFRDecoderTRT, MFREncoderTRT, DBNetTRT, CRNNTRT
from ._quiet import silence_mineru, suppress_c_stderr
from ._stagetimer import StageTimer


class FastMineruPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = (config or PipelineConfig()).resolve()
        self.config.apply_env()
        if self.config.quiet_mineru:
            silence_mineru()  # 静音 MinerU tqdm + loguru，rich 成为唯一进度层
        self._mfr_decoder: MFRDecoderTRT | None = None
        self._mfr_encoder: MFREncoderTRT | None = None
        self._dbnet: DBNetTRT | None = None
        self._crnn: CRNNTRT | None = None
        self._injected = False          # MFR-decoder
        self._mfr_enc_injected = False
        self._dbnet_injected = False
        self._crnn_injected = False
        self._ocr_gpu_injected = False
        self._ocr_det_gpu_injected = False
        self._whole_page_injected = False
        self._orig_batch_analyze = None
        self._mfr_gpu_injected = False

        rule("fast_mineru 初始化")
        with console.status("[cyan]加载模型 / 反序列化引擎 / 预分配 / warmup ...", spinner="dots"):
            self._t_init = time.perf_counter()
            self._load_models_and_warmup()   # 触发 MinerU atom model 创建(含 MFR head)
            self._load_and_inject_engines()  # 反序列化 TRT + 显式注入(非 patch)
            self._init_elapsed = time.perf_counter() - self._t_init

        self._print_banner()

    # ---- init 内部 ---------------------------------------------------------
    def _load_models_and_warmup(self):
        """用 batch_image_analyze 跑 warmup 页，一次性创建并预热全部 atom model。"""
        from PIL import Image as PILImage

        n = max(1, self.config.warmup_pages)
        dummy = [PILImage.new("RGB", (1024, 768), "white") for _ in range(n)]
        mb.warmup(
            [(img, True, self.config.lang) for img in dummy],
            formula_enable=self.config.formula_enable,
            table_enable=self.config.table_enable,
        )
        gc.collect()
        torch.cuda.empty_cache()

    def _get_mfr_head(self):
        """定位 pp_formulanet 的 head(PPFormulaNet_Head)。unimernet 无 generate_export 返回 None。"""
        return mb.get_mfr_head(self.config.device)

    def _iter_ocr_models(self):
        """遍历所有已创建的 OCR atom model 实例（覆盖 MinerU 建的所有 OCR 变体）。"""
        return mb.iter_ocr_models()

    def _load_and_inject_engines(self):
        """反序列化各 TRT 引擎并**一次性显式注入**(构造期一次，无 get_atom_model 拦截、无全局状态)。"""
        self._inject_mfr_decoder()
        self._inject_mfr_encoder()
        self._inject_dbnet()
        self._inject_crnn()
        self._inject_ocr_gpu()
        self._inject_ocr_det_gpu()
        self._inject_whole_page_gpu()
        self._inject_mfr_gpu()

    def _inject_whole_page_gpu(self):
        """把 pipeline_analyze 里 BatchAnalyze 名字重绑到 FastBatchAnalyze(整页 GPU 常驻)。
        需 CRNN TRT(rec 直通)才有意义；类级重绑一次，close() 还原。"""
        cfg = self.config
        if not cfg.use_whole_page_gpu:
            return
        if not self._crnn_injected:
            console.print("[yellow]⚠ 整页 GPU 常驻需 CRNN TRT，已跳过")
            return
        try:
            from .models.fast_batch_analyze import FastBatchAnalyze
        except Exception as e:
            console.print(f"[yellow]⚠ 整页 GPU 常驻不可用: {e}")
            return
        self._orig_batch_analyze = mb.install_fast_batch_analyze(FastBatchAnalyze)
        self._whole_page_injected = True
        console.print("[green]✓ 整页 GPU 常驻 → FastBatchAnalyze(OCR-det crop/BGR/mask 上 GPU, rec 直通)")

    def _inject_ocr_det_gpu(self):
        """把 OCR text_detector 的预处理提升为 GPU 版(csrc ocr_preprocess_image kernel)，
        消除 CPU DetResizeForTest。这是原版 --fast-ops 唯一加速的环节，追平原版的关键。"""
        cfg = self.config
        if not cfg.use_ocr_gpu_preprocess:
            return
        try:
            from .models.ocr import inject_ocr_det_gpu
        except Exception as e:
            console.print(f"[yellow]⚠ OCR-det GPU 预处理不可用(csrc?): {e}")
            return
        count = 0
        for ocr in self._iter_ocr_models():
            try:
                if inject_ocr_det_gpu(ocr):
                    count += 1
            except Exception:
                pass
        if count:
            self._ocr_det_gpu_injected = True
            console.print(f"[green]✓ OCR-det 预处理 → GPU csrc kernel({count} 实例)")

    def _inject_ocr_gpu(self):
        """把 OCR text_recognizer 的预处理提升为 GPU 版(csrc kernel)，消除逐 crop CPU resize。"""
        cfg = self.config
        if not cfg.use_ocr_gpu_preprocess:
            return
        try:
            from .models.ocr import inject_ocr_gpu
        except Exception as e:
            console.print(f"[yellow]⚠ OCR GPU 预处理不可用(csrc?): {e}")
            return
        count = 0
        for ocr in self._iter_ocr_models():
            try:
                if inject_ocr_gpu(ocr):
                    count += 1
            except Exception:
                pass
        if count:
            self._ocr_gpu_injected = True
            console.print(f"[green]✓ OCR-rec 预处理 → GPU csrc kernel({count} 实例)")

    def _inject_mfr_gpu(self):
        """把 FormulaRecognizer 的预处理(normalize+format+batch)提升为 GPU csrc kernel。
        crop_margin(PIL 内容感知)保留 CPU。构造期一次注入。"""
        cfg = self.config
        if not cfg.use_mfr_gpu_preprocess:
            return
        try:
            from .models.mfr import inject_mfr_gpu
            mfr = mb.get_mfr_model(cfg.device)
        except Exception as e:
            console.print(f"[yellow]⚠ MFR GPU 预处理不可用: {e}")
            return
        try:
            if inject_mfr_gpu(mfr):
                self._mfr_gpu_injected = True
                console.print("[green]✓ MFR 预处理 → GPU csrc kernel(1 实例)")
        except Exception as e:
            console.print(f"[yellow]⚠ MFR GPU 预处理注入失败: {e}")

    def _inject_mfr_decoder(self):
        cfg = self.config
        if not cfg.use_mfr_decoder_trt:
            return
        init_p, past_p = cfg.mfr_decoder_init_engine, cfg.mfr_decoder_past_engine
        if not (init_p.exists() and past_p.exists()):
            console.print(f"[yellow]⚠ MFR-decoder 引擎缺失，跳过: {init_p} / {past_p}")
            return
        head = self._get_mfr_head()
        if head is None:
            console.print("[yellow]⚠ 未找到 pp_formulanet head(是否 unimernet？)，跳过 MFR-decoder TRT")
            return
        self._mfr_decoder = MFRDecoderTRT(str(init_p), str(past_p), head=head, debug=cfg.debug)
        head._orig_generate_export = head.generate_export
        head.generate_export = self._mfr_decoder.generate_export
        head._fast_mineru_injected = True
        self._injected = True
        console.print(f"[green]✓ MFR-decoder → TensorRT(全程GPU零拷贝, max_batch={self._mfr_decoder.max_batch})")

    def _get_mfr_backbone(self):
        """定位 pp_formulanet 的 backbone(encoder) + DonutSwinModelOutput。返回 (backbone, cls) 或 (None,None)。"""
        return mb.get_mfr_backbone(self.config.device)

    def _inject_mfr_encoder(self):
        """把 pp_formulanet backbone(encoder，卷积、固定 [B,1,384,384])前向替换为 TRT。
        decoder 已单独 TRT；这里补上 encoder(MFR net 段 ~52% 耗时)。与 decoder 注入同构，构造期一次。"""
        cfg = self.config
        if not cfg.use_mfr_encoder_trt:
            return
        eng_p = cfg.mfr_encoder_engine
        if eng_p is None or not eng_p.exists():
            console.print(f"[yellow]⚠ MFR-encoder 引擎缺失，跳过: {eng_p}")
            return
        backbone, output_cls = self._get_mfr_backbone()
        if backbone is None:
            console.print("[yellow]⚠ 未找到 pp_formulanet backbone(是否 unimernet？)，跳过 MFR-encoder TRT")
            return
        if getattr(backbone.forward, "_fast_mineru_trt", False):
            self._mfr_enc_injected = True
            return
        try:
            self._mfr_encoder = MFREncoderTRT(str(eng_p), output_cls, debug=cfg.debug)
        except Exception as e:
            console.print(f"[yellow]⚠ MFR-encoder 引擎加载失败，跳过: {e}")
            return
        backbone._orig_forward = backbone.forward
        backbone.forward = self._mfr_encoder.make_wrapper(backbone.forward)
        self._mfr_enc_injected = True
        console.print(f"[green]✓ MFR-encoder(backbone) → TensorRT(decoder 仍走同一 TRT, max_batch={self._mfr_encoder.max_batch})")

    def _inject_dbnet(self):
        cfg = self.config
        if not cfg.use_dbnet_trt or not cfg.dbnet_engine.exists():
            if cfg.use_dbnet_trt:
                console.print(f"[yellow]⚠ DBNet 引擎缺失，跳过: {cfg.dbnet_engine}")
            return
        self._dbnet = DBNetTRT(str(cfg.dbnet_engine))
        # 包所有已建 OCR 实例的 det.net.forward(按 net 身份去重，避免共享 net 被包两次)。
        # TRT 出 maps 后包成 {'maps':...}，与 _build_det_preds 契约一致。
        seen, count = set(), 0
        for ocr in self._iter_ocr_models():
            det = getattr(ocr, "text_detector", None)
            net = getattr(det, "net", None) if det is not None else None
            if net is None or id(net) in seen or hasattr(net, "_orig_forward"):
                continue
            seen.add(id(net))
            net._orig_forward = net.forward
            net.forward = self._dbnet.wrap_forward(net.forward)
            count += 1
        if count:
            self._dbnet_injected = True
            console.print(f"[green]✓ DBNet(OCR-det) → TensorRT({count} 实例, max_batch={self._dbnet.max_batch})")
        else:
            console.print("[yellow]⚠ 未找到 TextDetector.net，跳过 DBNet TRT")

    def _inject_crnn(self):
        cfg = self.config
        if not cfg.use_crnn_trt or not cfg.crnn_engine.exists():
            if cfg.use_crnn_trt:
                console.print(f"[yellow]⚠ CRNN 引擎缺失，跳过: {cfg.crnn_engine}")
            return
        self._crnn = CRNNTRT(str(cfg.crnn_engine))
        seen, count = set(), 0
        for ocr in self._iter_ocr_models():
            rec = getattr(ocr, "text_recognizer", None)
            net = getattr(rec, "net", None) if rec is not None else None
            if net is None or id(net) in seen or hasattr(net, "_orig_forward"):
                continue
            seen.add(id(net))
            net._orig_forward = net.forward
            net.forward = self._crnn.wrap_forward(net.forward)
            count += 1
        if count:
            self._crnn_injected = True
            console.print(f"[green]✓ CRNN(OCR-rec) → TensorRT({cfg.crnn_engine_precision}, {count} 实例, max_batch={self._crnn.max_batch})")
        else:
            console.print("[yellow]⚠ 未找到 TextRecognizer.net，跳过 CRNN TRT")

    def _print_banner(self):
        cfg = self.config
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" \
            if torch.cuda.is_available() else "-"
        yn = lambda b: "✓ 已注入" if b else "✗ (torch)"
        kv_panel("fast_mineru 就绪", {
            "device": dev,
            "VRAM": vram,
            "MFR-decoder TRT": yn(self._injected),
            "MFR-encoder TRT": yn(self._mfr_enc_injected),
            "MFR 预处理 GPU": yn(self._mfr_gpu_injected),
            "DBNet(det) TRT": yn(self._dbnet_injected),
            "CRNN(rec) TRT": yn(self._crnn_injected),
            "OCR-rec GPU 预处理": yn(self._ocr_gpu_injected),
            "OCR-det GPU 预处理": yn(self._ocr_det_gpu_injected),
            "整页 GPU 常驻": yn(self._whole_page_injected),
            "MFR precision": cfg.mfr_precision,
            "formula model": "pp_formulanet_plus_m" if cfg.formula_ch_support else "unimernet_small",
            "init 耗时": f"{self._init_elapsed:.1f}s",
        }, style="green")

    # ---- process：纯推理 ---------------------------------------------------
    def process(self, pdf_path: str | Path) -> dict:
        """对单篇 PDF 做推理。无加载/无分配。返回 {name, output_dir, timing, process_ms, pages}。"""
        pdf_path = Path(pdf_path)
        from .mineru_backend import _process_output, prepare_env, FileBasedDataWriter

        cfg = self.config
        timer = Timer()
        stage_timer = StageTimer(deep=cfg.debug).install() if cfg.stage_timing else None
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.stem

        local_image_dir, local_md_dir = prepare_env(str(cfg.output_dir), pdf_name, cfg.parse_method)
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        from .mineru_backend import MakeMode

        render = not cfg.no_render  # no_render: 跳过画框PDF/原PDF/markdown 渲染(不可加速的重活)

        def on_doc_ready(doc_index, model_list, middle_json, ocr_enable):
            _process_output(
                pdf_info=middle_json["pdf_info"], pdf_bytes=pdf_bytes, pdf_file_name=pdf_name,
                local_md_dir=local_md_dir, local_image_dir=local_image_dir, md_writer=md_writer,
                f_draw_layout_bbox=render, f_draw_span_bbox=render, f_dump_orig_pdf=render,
                f_dump_md=render, f_dump_content_list=True, f_dump_middle_json=True,
                f_dump_model_output=render, f_make_md_mode=MakeMode.MM_MD,
                middle_json=middle_json, model_output=model_list, process_mode="pipeline",
            )

        # 仅"渲染 + quiet"时才 fd 级抑制 pdfium 的 /Ascent 噪音；no_render 无渲染则不动 fd
        # (避免多余 fd 重定向，也彻底消除与 loguru 的窗口冲突)。
        stderr_guard = (suppress_c_stderr() if (cfg.quiet_mineru and not cfg.no_render)
                        else contextlib.nullcontext())
        # 每篇前关闭上次可能残留的 render workers(Windows spawn 模式下继承 CUDA
        # context 极易触发 VRAM OOM)，并重试一次 BrokenProcessPool。
        mb.shutdown_render_executor()

        t0 = time.perf_counter()
        with timer.section("analyze"), stderr_guard:
            for attempt in range(2):
                try:
                    mb.analyze_streaming(
                        pdf_bytes_list=[pdf_bytes], image_writer_list=[image_writer],
                        lang_list=[cfg.lang], on_doc_ready=on_doc_ready,
                        parse_method=cfg.parse_method,
                        formula_enable=cfg.formula_enable, table_enable=cfg.table_enable,
                    )
                except RuntimeError as e:
                    if "BrokenProcessPool" in str(e) and attempt == 0:
                        mb.shutdown_render_executor()
                        continue
                    raise
                else:
                    break
        process_ms = (time.perf_counter() - t0) * 1000
        stage_rows = stage_timer.rows(process_ms) if stage_timer else None
        if stage_timer:
            stage_timer.uninstall()

        pages = self._count_pages(pdf_bytes)
        return {
            "stage_rows": stage_rows,
            "name": pdf_name,
            "output_dir": str(local_md_dir),
            "timing": timer,
            "process_ms": process_ms,
            "pages": pages,
        }

    def process_many(self, paths: list) -> list:
        """多文档批处理，复用同一组模型/引擎。"""
        results = []
        for p in paths:
            results.append(self.process(p))
        return results

    @staticmethod
    def _count_pages(pdf_bytes: bytes) -> int:
        return mb.count_pages(pdf_bytes)

    # ---- 清理 --------------------------------------------------------------
    def close(self):
        """恢复所有被注入的方法(可选)。"""
        if self._injected:
            head = self._get_mfr_head()
            if head is not None and getattr(head, "_fast_mineru_injected", False):
                head.generate_export = head._orig_generate_export
                head._fast_mineru_injected = False
            self._injected = False
        if self._mfr_enc_injected:
            backbone, _ = self._get_mfr_backbone()
            if backbone is not None and hasattr(backbone, "_orig_forward"):
                backbone.forward = backbone._orig_forward
                del backbone._orig_forward
            self._mfr_enc_injected = False
        if self._mfr_gpu_injected:
            from .models.mfr import restore_mfr_gpu
            try:
                restore_mfr_gpu(mb.get_mfr_model(self.config.device))
            except Exception:
                pass
            self._mfr_gpu_injected = False
        if self._dbnet_injected or self._crnn_injected:
            for ocr in self._iter_ocr_models():
                for attr in ("text_detector", "text_recognizer"):
                    sub = getattr(ocr, attr, None)
                    net = getattr(sub, "net", None) if sub is not None else None
                    if net is not None and hasattr(net, "_orig_forward"):
                        net.forward = net._orig_forward
                        del net._orig_forward
            self._dbnet_injected = self._crnn_injected = False
        if self._ocr_gpu_injected:
            from .models.ocr import restore_ocr_gpu
            for ocr in self._iter_ocr_models():
                restore_ocr_gpu(ocr)
            self._ocr_gpu_injected = False
        if self._ocr_det_gpu_injected:
            from .models.ocr import restore_ocr_det_gpu
            for ocr in self._iter_ocr_models():
                restore_ocr_det_gpu(ocr)
            self._ocr_det_gpu_injected = False
        if self._whole_page_injected:
            mb.restore_batch_analyze(self._orig_batch_analyze)
            self._whole_page_injected = False
