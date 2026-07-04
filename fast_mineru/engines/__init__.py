"""fast_mineru.engines — TensorRT 引擎封装(纯类，零拷贝，无 monkey-patch)。"""
from .trt_base import TRTEngine
from .mfr_decoder import MFRDecoderTRT
from .dbnet import DBNetTRT
from .crnn import CRNNTRT

__all__ = ["TRTEngine", "MFRDecoderTRT", "DBNetTRT", "CRNNTRT"]
