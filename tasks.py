"""invoke 工作流：编译 csrc / 打 wheel / 生成 .pyi 存根 / 跑基准。

用法(uv 环境)：
    uv run invoke build-ext      # cmake 配置+编译 _fast_mineru_core.pyd
    uv run invoke gen-stub       # 从 .pyd 生成类型存根 .pyi
    uv run invoke build          # uv build 打 wheel + retag 平台标签
    uv run invoke bench -p xxx.pdf
"""
from invoke import task
import sys
import tomllib

# Windows 下 invoke 转发子进程输出用 GBK，nvcc/cmake 的 UTF-8 输出会 UnicodeEncodeError。
# 重配 stdout/stderr 为 UTF-8(errors=replace 兜底)。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _version():
    with open("pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


@task(name="build-ext")
def build_ext(c, config="Release", clean=False):
    """cmake 配置 + 编译 CUDA pybind11 模块 → fast_mineru/csrc/_fast_mineru_core.pyd。"""
    build_dir = "_build"
    if clean:
        import shutil, os
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
    c.run(f'cmake -S . -B {build_dir} -DCMAKE_BUILD_TYPE={config}')
    c.run(f'cmake --build {build_dir} --config {config} --target _fast_mineru_core')
    print("[fast_mineru] csrc 编译完成 → fast_mineru/csrc/")


@task
def retag(c, version=None):
    """把 uv build 产出的 any-wheel 重打成本平台标签(cp312)。"""
    version = version or _version()
    if sys.platform == "win32":
        platform_tag = "win_amd64"
    elif sys.platform == "linux":
        platform_tag = "manylinux_2_17_x86_64"
    elif sys.platform == "darwin":
        platform_tag = "macosx_10_15_x86_64"
    else:
        raise RuntimeError(f"不支持的平台: {sys.platform}")
    c.run(
        f"uv run python -m wheel tags "
        f"--python-tag cp312 --abi-tag cp312 --platform-tag {platform_tag} "
        f"./dist/fast_mineru-{version}-py3-none-any.whl"
    )


@task
def build(c):
    """打 wheel(先确保 csrc 已编译)。"""
    c.run("uv build")
    retag(c)
    print("[fast_mineru] 构建完成")


@task(name="gen-stub")
def gen_stub(c):
    """从 _fast_mineru_core.pyd 生成 .pyi 存根(IDE 补全 / 类型检查)。

    前置：先 build-ext 编出可导入的 .pyd。
    """
    import shutil
    from pathlib import Path

    module = "fast_mineru.csrc._fast_mineru_core"
    tmp = Path("_stubgen_tmp")
    dst = Path("fast_mineru/csrc/_fast_mineru_core.pyi")

    if tmp.exists():
        shutil.rmtree(tmp)
    c.run(
        'uv run --no-sync python -m pybind11_stubgen '
        '--ignore-invalid=all --root-module-suffix "" '
        f'-o {tmp} {module}'
    )
    src = tmp / "fast_mineru" / "csrc" / "_fast_mineru_core" / "__init__.pyi"
    if not src.exists():
        src = tmp / "fast_mineru" / "csrc" / "_fast_mineru_core.pyi"
    if not src.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"未找到生成的存根(.pyd 是否已编译并可导入？)")
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"已生成类型存根: {dst}")


@task
def bench(c, p, output="fast_mineru_out"):
    """端到端基准：跑 CLI 并打印计时表。"""
    c.run(f'uv run fast-mineru "{p}" --output "{output}" --bench')
