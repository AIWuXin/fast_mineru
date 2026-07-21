"""PipelineConfig —— 所有加速开关 / 引擎路径 / 精度的单一数据类。

替代源项目散落的十几个 env(MFR_INFERENCE_PRECISION / MFR_DEC_DEBUG /
MINERU_FORMULA_CH_SUPPORT / ...)。配置驱动，不用运行时全局变量。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 引擎默认目录：包内 engines_bin/ 优先，其次项目根。
_PKG_ROOT = Path(__file__).resolve().parent
_DEFAULT_ENGINE_DIR = _PKG_ROOT.parent / "engines_bin"


@dataclass
class PipelineConfig:
    # ---- 加速开关 ----
    use_mfr_decoder_trt: bool = True   # ★ 核心：MFR decoder 两段式 TRT(全程GPU零拷贝)
    use_mfr_encoder_trt: bool = True   # encoder(卷积,固定[B,1,384,384]) → TRT，MFR net 段 ~52% 耗时
    use_dbnet_trt: bool = True         # OCR-det TensorRT
    use_crnn_trt: bool = True          # OCR-rec TensorRT
    use_fast_ops: bool = True          # OCR-det 预处理 CUDA kernel
    use_ocr_gpu_preprocess: bool = True  # OCR-rec 逐 crop resize 走 GPU csrc kernel(替代 CPU cv2)
    use_whole_page_gpu: bool = False   # 整页 GPU 常驻(FastBatchAnalyze)。实测(2026-07-21, mineru 3.4.1):
                                       # 逐页小批 det+inline rec + 逐行 Python 透视裁剪,比原生编排
                                       # 慢 ~40%(27.5s vs 19s/doc)。显存尖峰已由 rec 宽度预算分批根治,
                                       # 不需要靠它压显存。默认关;--whole-page-gpu 可开(实验性)。
    use_mfr_gpu_preprocess: bool = True  # MFR normalize+format+batch → csrc kernel(保留 CPU crop_margin)
    use_torch_rec: bool = False        # 诊断开关：OCR-rec 整段回退 mineru 原生 torch(CPU resize_norm + torch CRNN)。
                                       # 跳过 CRNN TRT + rec GPU 预处理注入，FastBatchAnalyze 给 rec 产出 CPU numpy crop。
                                       # det/layout/MFR 仍 TRT+GPU 不变 —— 用于隔离验证 rec 是否为显存锯齿来源。

    # ---- 引擎路径(None → 用 engine_dir 下默认名) ----
    engine_dir: Path = _DEFAULT_ENGINE_DIR
    mfr_decoder_init_engine: Path | None = None
    mfr_decoder_past_engine: Path | None = None
    mfr_encoder_engine: Path | None = None
    dbnet_engine: Path | None = None
    crnn_engine: Path | None = None

    # ---- 精度(构造参数，非 env) ----
    mfr_precision: str = "fp16"        # fp16 / fp32；TRT decoder 内部 fp16 已对齐 fp32 金标准
    ocr_precision: str = "auto"        # MinerU torch-OCR 精度门控：auto / fp32 / fp16(不含 tf32)
    crnn_engine_precision: str = "tf32"  # CRNN **TRT 引擎文件**选择：tf32(高精度) / fp16(快)。与 ocr_precision 无关

    # ---- 模型行为 ----
    formula_ch_support: bool = True    # pp_formulanet_plus_m(高精度公式模型)
    formula_enable: bool = True
    table_enable: bool = True
    parse_method: str = "auto"         # auto / ocr / txt
    lang: str = "ch"

    # ---- 运行 ----
    device: str = "cuda"
    warmup_pages: int = 2              # init 期 warmup 页数(固化 cudnn/TRT tactic)
    pdf_render_processes: int = 8      # pdfium 光栅化进程数(mineru 硬编码上限 3;20 核机器放开到 8)。
                                       # 需配合路径版加载器(免每任务 112MB pickle IPC),0=保持 mineru 默认。
    prefetch_render: bool = True       # 流式编排(streaming.py):GPU analyze 当前窗口时预取渲染下一窗口。
                                       # 实测渲染等待占 29%,预取后几乎全隐藏。
    overlap_append: bool = True        # 流式编排:逐页后处理(MagicModel/middle_json,~16%)与下一窗口
                                       # analyze 重叠(单线程保序,finalize 时机不变,输出逐字节一致)。
    output_workers: int = 4            # 输出进程池(pypdf 画框/md 落盘)worker 数;缩短多文档尾部排空。
    clean_cache_threshold_gb: float = 7.0  # 窗口末 empty_cache 阈值(reserved 超过才真清);
                                       # <=0 保持 mineru 原版每窗口全量清。
    low_priority_workers: bool = False # 渲染 worker 进程 + append 线程降优先级。实测(2026-07-21):
                                       # 系统并未饱和(~14 活跃/20 核),降优先级只拖慢 helper 反而 -4%。
                                       # stage wall 膨胀的真因是 GIL(append 线程),OS 优先级够不到。
    prefetch_page_chars: bool = False  # 渲染 worker 顺带提取页面文本层 chars(get_page_chars
                                       # ~120ms/页),append 线程查缓存。实测(2026-07-21):GIL 减负
                                       # (-6.5s stage wall)≈ 渲染路径增重(+9s IPC/任务变重),净收益≈0。
                                       # 留作实验开关;要根治需更快的 C 级字符提取。
    debug: bool = False                # 打印每组 TRT 计时(替代 MFR_DEC_DEBUG)
    stage_timing: bool = False         # 细分各模型 stage wall(layout/mfr/table/det/rec + 其它)
    output_dir: Path = field(default=Path("fast_mineru_out"))

    # ---- 输出 ----
    no_render: bool = False            # 跳过 markdown/画框PDF 渲染(纯测推理速度用)。
                                       # 只 dump middle_json/content_list，消除 PDF 渲染这段不可加速开销
                                       # + 顺带消除 pdfium 渲染的 /Ascent stderr 噪音。
    quiet_mineru: bool = True          # 静音 MinerU 内部 tqdm/日志噪音，只留 fast_mineru 的 rich 输出

    def resolve(self) -> "PipelineConfig":
        """把 None 引擎路径补成 engine_dir 下默认名，并规范为绝对 Path。"""
        d = Path(self.engine_dir)
        defaults = {
            "mfr_decoder_init_engine": "decoder_init_fp16.engine",
            "mfr_decoder_past_engine": "decoder_with_past_fp16.engine",
            "mfr_encoder_engine": "encoder_ppformulanet_fp16.engine",
            "dbnet_engine": "dbnet.engine",
            "crnn_engine": "crnn_fp16.engine" if self.crnn_engine_precision == "fp16" else "crnn.engine",
        }
        for attr, name in defaults.items():
            if getattr(self, attr) is None:
                setattr(self, attr, d / name)
            else:
                setattr(self, attr, Path(getattr(self, attr)))
        self.output_dir = Path(self.output_dir)
        return self

    def apply_env(self):
        """把必须走 env 的 MinerU 门控落地(仅这几个 MinerU 源码强依赖 env 的开关)。"""
        os.environ["MINERU_FORMULA_CH_SUPPORT"] = "True" if self.formula_ch_support else "False"
        os.environ["MFR_INFERENCE_PRECISION"] = self.mfr_precision
        os.environ["OCR_INFERENCE_PRECISION"] = self.ocr_precision
