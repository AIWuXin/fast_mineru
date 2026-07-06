# fast_mineru

MinerU 推理管线的**原生加速封装**：一个 `FastMineruPipeline` 类，init 加载/预分配/warmup，`process()` 纯推理。

核心加速：**MFR decoder 两段式 TensorRT，全程 GPU 零拷贝**(无 H2D/D2H) + CUDA 预处理算子(csrc)。
精度实测 TRT decoder **160/160 对齐 fp32 金标准**(甚至比 torch-autocast-fp16 158/160 更准)。

> **依赖 PyPI 原版 `mineru[pipeline]==3.4.1`**，不再需要相邻的 MinerU 源码树 —— 克隆本仓库即可从零构建运行。
> 所有对 mineru 的引用收敛在单一适配器 `fast_mineru/mineru_backend.py`，加速逻辑不依赖任何 mineru 魔改。

## 快速开始(uv 管理环境) —— 克隆后从零到能跑

```bash
# 1. 装环境(mineru==3.4.1 走 PyPI，torch 走 cu128 explicit index，见 pyproject.toml)
uv sync

# 2. 编译 csrc CUDA 算子 → fast_mineru/csrc/_fast_mineru_core.pyd
uv run invoke build-ext

# 3. 构建 5 个 TensorRT 引擎 → engines_bin/(首次自动下载 mineru 权重)
#    流程：架构自检 → 导出 ONNX → trtexec 编译。engine 是大文件、绑 GPU/TRT 版本，不入 git，本地构建。
uv run invoke build-engines                        # crnn 默认 tf32(逐字对齐 CPU)
#    uv run invoke build-engines --crnn both        # 同时构建 tf32+fp16 的 crnn

# 4.(可选)生成类型存根
uv run invoke gen-stub

# 5. 跑推理：单 PDF 或文件夹(自动递归多文档)
uv run fast-mineru paper.pdf --output out --bench
uv run fast-mineru ./papers/ --output out          # 文件夹批处理，模型只加载一次
```

> Python **定死 CPython 3.12**(`requires-python = "==3.12.*"`)。`uv.lock` 入库保证可复现构建。
> 前置：CUDA GPU + TensorRT 10.x(`trtexec` 随 pip 的 tensorrt 包提供，构建脚本自动定位)。

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
  mineru_backend.py ★ 与 mineru 的唯一接缝(懒再导出全部符号 + 单例句柄 helper)
  engines/
    trt_base.py    TRTEngine 基类(专用 stream, data_ptr 零拷贝, execute_async_v3)
    mfr_decoder.py MFRDecoderTRT(init+past 两引擎, 贪心循环, B>max 拆块)
  models/          加速感知模型封装(OCR/MFR 预处理 GPU + FastBatchAnalyze 整页常驻)
  csrc/            ★ CUDA kernel(pybind11): ocr_preprocess + binding → _fast_mineru_core.pyd
engine_build/      ★ 引擎构建链(路径无关): _env(权重/trtexec 定位) + export_*.py + build_all.py
tools/             cmp_all_latex.py(精度对比)
engines_bin/       构建产物 .engine(不入 git, 由 engine_build/ 生成)
tests/             pytest
```

## 引擎构建细节

`invoke build-engines` 会构建与 `config.py::resolve()` 默认名对齐的 5 个引擎：

| 引擎 | ONNX 来源脚本 | trtexec 精度 |
|---|---|---|
| `decoder_init_fp16.engine` / `decoder_with_past_fp16.engine` | `export_decoder_*_onnx.py` | fp16(内部已对齐 fp32 金标准) |
| `encoder_ppformulanet_fp16.engine` | `export_encoder_onnx.py` | fp16 |
| `dbnet.engine` | `export_det_rec_onnx.py` | fp16(bbox 逐像素一致) |
| `crnn.engine`(默认) / `crnn_fp16.engine` | `export_det_rec_onnx.py` | tf32(逐字对齐 CPU) / fp16(快, 极小抖动) |

单独构建某个引擎：`uv run --no-sync python engine_build/build_engines.py --only encoder,dbnet`。

