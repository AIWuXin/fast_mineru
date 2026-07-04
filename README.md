# fast_mineru

MinerU 推理管线的**原生加速封装**：把 `D:/project/MinerU` 里 fast_ops 的 monkey-patch 加速工作，
重写成**一等公民**的干净架构 —— 一个 `FastMineruPipeline` 类，init 加载/预分配/warmup，`process()` 纯推理。

核心加速：**MFR decoder 两段式 TensorRT，全程 GPU 零拷贝**(无 H2D/D2H) + CUDA 预处理算子(csrc)。
精度实测 TRT decoder **160/160 对齐 fp32 金标准**(甚至比 torch-autocast-fp16 158/160 更准)。

## 快速开始(uv 管理环境)

```bash
# 1. 装环境(torch 走 cu128 explicit index，见 pyproject.toml)
uv sync

# 2. 编译 csrc CUDA 算子 → fast_mineru/csrc/_fast_mineru_core.pyd
uv run invoke build-ext

# 3.(可选)生成类型存根
uv run invoke gen-stub

# 4. 跑推理：单 PDF 或文件夹(自动递归多文档)
uv run fast-mineru paper.pdf --output out --bench
uv run fast-mineru ./papers/ --output out          # 文件夹批处理，模型只加载一次
```

> Python **定死 CPython 3.12**(`requires-python = "==3.12.*"`)：环境稳定，复用既有构建缓存。

## 代码方式

```python
from fast_mineru import FastMineruPipeline, PipelineConfig

pipe = FastMineruPipeline(PipelineConfig(output_dir="out"))  # 一次性加载+预分配+warmup
r = pipe.process("paper.pdf")                                # 纯推理，无加载/无分配
print(r["process_ms"], r["output_dir"])
pipe.process_many(["a.pdf", "b.pdf"])                        # 复用同一组模型
```

`__init__` 完成一切前置(模型加载、TRT 引擎反序列化、**显式注入** decoder TRT 到 head、warmup)；
`process()` 只做 读取→前向→后处理，返回结构化结果 + 每 stage 计时 + **process 总耗时**。
CLI 结尾区分 **init 耗时(一次)** 与 **process 总耗时 / 平均 / pages·s**。

## 架构

```
fast_mineru/
  config.py        PipelineConfig 数据类(所有加速开关/引擎路径/精度，替代散落 env)
  console.py       rich Console 单例 + 计时表/面板/Timer
  pipeline.py      ★ FastMineruPipeline(init 预分配, process 零加载, 构造期显式注入)
  cli.py           ★ CLI: 单 PDF/文件夹, rich 输出, process 总耗时
  engines/
    trt_base.py    TRTEngine 基类(专用 stream, data_ptr 零拷贝, execute_async_v3)
    mfr_decoder.py MFRDecoderTRT(init+past 两引擎, 贪心循环, B>max 拆块)
  models/          加速感知模型封装(未来把注入下沉为 FormulaRecognizer 子类)
  csrc/            ★ CUDA kernel(pybind11): ocr_preprocess + binding → _fast_mineru_core.pyd
build/             引擎/ONNX 构建脚本(build_decoder_engines.py, export_*.py)
tools/             cmp_all_latex.py(精度对比)
engines_bin/       预转好的 .engine
tests/             pytest
```
