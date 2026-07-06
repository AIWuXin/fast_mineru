# -*- coding: utf-8 -*-
"""engine_build 的可移植环境定位 —— 消灭硬编码路径，克隆到任何机器可直接跑。

三件事：
1. **权重目录**：复刻 mineru 内部 model_init 的构造方式，自动下载并定位 pp_formulanet_plus_m
   权重（modelscope/huggingface，随 MINERU_MODEL_SOURCE），不再硬编码 `C:/Users/.../modelscope/...`。
2. **trtexec**：PATH 优先，回退到已安装的 tensorrt / tensorrt_libs 包目录，跨平台。
3. **输出目录**：onnx 与 engine 统一落到项目 `engines_bin/`（config.py::resolve() 默认读这里），
   不再写死 `D:/project/MinerU/`。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

# 项目根 = engine_build/ 的父目录；引擎/ONNX 统一落 engines_bin/。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PROJECT_ROOT / "engines_bin"

OPSET = 17            # 全部 onnx 导出统一 opset（decoder/encoder/det/rec 一致）
N_LAYERS = 6          # pp_formulanet decoder 层数
CROSS = 144           # encoder 输出序列长度（cross-attn K/V 的固定维）


def get_mfr_weight_dir() -> str:
    """定位 pp_formulanet_plus_m 权重目录（缺失则自动下载）。

    1:1 复刻 mineru/backend/pipeline/model_init.py 的构造：
        os.path.join(auto_download_and_get_model_root_path(rel), rel)
    这样与 mineru 运行时加载的是同一份权重，跨机器可移植。
    """
    os.environ.setdefault("MINERU_FORMULA_CH_SUPPORT", "True")
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
    from mineru.utils.enum_class import ModelPath
    rel = ModelPath.pp_formulanet_plus_m
    return str(os.path.join(auto_download_and_get_model_root_path(rel), rel))


def find_trtexec() -> str:
    """定位 trtexec 可执行文件：PATH → tensorrt/tensorrt_libs 包目录。跨平台。"""
    exe = shutil.which("trtexec")
    if exe:
        return exe
    for pkg in ("tensorrt_libs", "tensorrt"):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            spec = None
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
        for loc in locs:
            for name in ("trtexec.exe", "trtexec"):
                cand = Path(loc) / name
                if cand.exists():
                    return str(cand)
    raise FileNotFoundError(
        "未找到 trtexec：确认已安装 tensorrt（pip 包自带 trtexec），或将 trtexec 加入 PATH。"
    )


def ensure_engines_dir() -> Path:
    """确保 engines_bin/ 存在并返回。"""
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    return ENGINES_DIR


def onnx_path(name: str) -> str:
    """engines_bin/ 下的 onnx 路径（字符串）。"""
    return str(ensure_engines_dir() / name)


def engine_path(name: str) -> str:
    """engines_bin/ 下的 engine 路径（字符串）。"""
    return str(ensure_engines_dir() / name)
