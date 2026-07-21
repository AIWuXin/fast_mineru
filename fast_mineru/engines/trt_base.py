"""TRTEngine —— TensorRT 10 引擎基类：反序列化 + 专用 stream 零拷贝执行。

设计要点(来自 D:/project/MinerU 的血泪经验)：
- **全程 GPU 零拷贝**：输入直接绑 torch cuda tensor 的 data_ptr，输出预建 torch cuda tensor
  绑 data_ptr，execute_async_v3 后无任何 H2D/D2H。
- **专用非默认 stream**：避免 "Using default stream in enqueueV3()" 告警 + 额外同步；
  用 wait_stream 双向定序(输入就绪→TRT→输出就绪)，record_stream 保证 caching allocator
  不提前回收输出 buffer。精度不变。
"""
from __future__ import annotations

import threading

import torch

# 所有 TRT 引擎共享的全局执行锁。IExecutionContext 不是线程安全的；流式编排
# (streaming.py)的 append 线程会在 finalize 里跑 post-OCR rec(走 CRNN TRT),
# 与主线程 analyze 的引擎执行并发。TRT 在 GPU 上本就串行执行,一把全局锁
# 跨引擎互斥无性能损失(锁内只有 enqueue,纳秒级开销),只保证线程安全。
_EXEC_LOCK = threading.Lock()


class TRTEngine:
    """单个 TRT 引擎的薄封装。子类/调用方通过 run(feed) 拿 name->cuda tensor 输出。"""

    def __init__(self, engine_path: str, stream: torch.cuda.Stream | None = None):
        import tensorrt as trt
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"反序列化 TRT 引擎失败: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs = self._io_names(self.engine)
        # 复用外部传入的专用 stream(多引擎共享同一条流可省去跨流同步)，否则自建。
        self.stream = stream if stream is not None else torch.cuda.Stream()
        self.path = engine_path

    def _io_names(self, engine):
        trt = self._trt
        ins, outs = [], []
        for i in range(engine.num_io_tensors):
            nm = engine.get_tensor_name(i)
            if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                ins.append(nm)
            else:
                outs.append(nm)
        return ins, outs

    def profile_batch_max(self, tensor_name: str = "input_ids", profile: int = 0) -> int:
        """从引擎 profile 读某输入的 batch 上限(maxShapes 的第 0 维)。"""
        _, _, mx = self.engine.get_tensor_profile_shape(tensor_name, profile)
        return int(mx[0])

    def run(self, feed: dict, out_buffers: dict | None = None) -> dict:
        """feed: name->torch cuda tensor。返回 name->torch cuda tensor(fp32 输出)。

        out_buffers: 可选，name->预分配 cuda tensor(减少每步 torch.empty)。形状不匹配则新建。
        线程安全：全局执行锁互斥(见模块顶 _EXEC_LOCK)。
        """
        with _EXEC_LOCK:
            return self._run_locked(feed, out_buffers)

    def _run_locked(self, feed: dict, out_buffers: dict | None = None) -> dict:
        trt = self._trt
        engine, ctx = self.engine, self.ctx
        for nm, t in feed.items():
            ctx.set_input_shape(nm, tuple(t.shape))
        outs = {}
        for i in range(engine.num_io_tensors):
            nm = engine.get_tensor_name(i)
            if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                ctx.set_tensor_address(nm, feed[nm].data_ptr())
            else:
                shp = tuple(ctx.get_tensor_shape(nm))
                o = None
                if out_buffers is not None:
                    buf = out_buffers.get(nm)
                    if buf is not None and tuple(buf.shape) == shp:
                        o = buf
                if o is None:
                    o = torch.empty(shp, dtype=torch.float32, device="cuda")
                outs[nm] = o
                ctx.set_tensor_address(nm, o.data_ptr())
        # 专用流执行：双向 wait_stream 定序，无全量同步。
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        ctx.execute_async_v3(self.stream.cuda_stream)
        cur.wait_stream(self.stream)
        for o in outs.values():
            o.record_stream(self.stream)
        return outs
