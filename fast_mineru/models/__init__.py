"""fast_mineru.models —— 加速感知的模型封装(未来重构点)。

当前 MFR-decoder TRT 通过 FastMineruPipeline 在构造期**显式注入** head.generate_export
(见 pipeline.py，非 monkey-patch，一次绑定)。

后续可把注入进一步下沉为 FormulaRecognizer 子类：在 __init__ 里按 config 决定
self.net.head.generate_export = MFRDecoderTRT(...).generate_export，让加速成为模型类型
本身的一部分，彻底不碰 MinerU 全局。留作 §4 架构的下一步。

ocr.py：FastTextRecognizer/FastTextDetector(OCR rec+det GPU 预处理覆写) +
inject_ocr_gpu/restore_ocr_gpu + inject_ocr_det_gpu/restore_ocr_det_gpu。
"""
from .ocr import (
    inject_ocr_det_gpu,
    inject_ocr_gpu,
    restore_ocr_det_gpu,
    restore_ocr_gpu,
)

from .fast_batch_analyze import FastBatchAnalyze

__all__ = [
    "inject_ocr_gpu", "restore_ocr_gpu",
    "inject_ocr_det_gpu", "restore_ocr_det_gpu",
    "FastBatchAnalyze",
]
