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

**无 monkey-patch**：`FastMineruPipeline.__init__` 里按 config 把 MFR head 的 `generate_export`
一次性绑成 TRT 版，不靠运行时 `get_atom_model` 拦截、不用全局状态。

## 混合项目构建(照搬 nemotron_infer)

- `CMakeLists.txt`：`pybind11_add_module` + `CMAKE_CUDA_ARCHITECTURES 89 120`(RTX40/50) +
  POST_BUILD 把 `.pyd` 和 `cudart64_*.dll` 拷进 `fast_mineru/csrc/`。
- `tasks.py`(invoke)：`build-ext` 编 csrc、`build` 打 wheel + `retag` 平台标签、`gen-stub` 出 `.pyi`。
- MSVC `/utf-8`，CUDA `--expt-relaxed-constexpr`。

## 血泪教训(内化自源项目)

- **别信隔离基准**：encoder 隔离 4.8x → 端到端仅 9%；batch=1 decoder "3.8x" 是 launch-bound 假象。
  必须端到端实测占比。
- **B>引擎 max_batch 必须拆块**(48→32+16 pad 拼接)，且不能在拆块前早退回 torch(源项目踩过死代码坑)。
- **全程 GPU 零拷贝 + 专用非默认 stream**：present KV 直接 data_ptr 当下一步 past，消除
  "Using default stream in enqueueV3" 告警且精度不变。

详见 `../.context/fast_mineru_blueprint.md`。
