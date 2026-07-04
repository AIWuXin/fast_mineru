"""fast_mineru —— MinerU 推理管线的原生加速封装。

- FastMineruPipeline：一个类，init 加载/预分配/warmup，process() 纯推理。
- 加速：MFR decoder 两段式 TensorRT(全程 GPU 零拷贝) + CUDA 预处理算子(csrc)。
- 无 monkey-patch：加速作为一等公民，构造期显式注入。

用法：
    from fast_mineru import FastMineruPipeline, PipelineConfig
    pipe = FastMineruPipeline(PipelineConfig(output_dir="out"))
    result = pipe.process("paper.pdf")
    print(result["process_ms"])
"""
from .config import PipelineConfig
from .pipeline import FastMineruPipeline

__version__ = "0.1.0"
__all__ = ["FastMineruPipeline", "PipelineConfig", "__version__"]
