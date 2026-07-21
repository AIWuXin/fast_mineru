"""FastTextRecognizer / FastTextDetector —— OCR 子模型的 GPU 预处理覆写(方案 B)。

不 fork BatchAnalyze.__call__，而是子类化 MinerU 的 TextRecognizer/TextDetector 覆写其自有
方法，构造期把实例 __class__ 提升为子类(干净的实例级注入，对 __call__ 变动免疫)。
计算部分走我们的 csrc kernel + TRT；非计算部分(后处理/过滤)直接调 MinerU 原方法。

- FastTextRecognizer.__call__：把逐 crop 的 CPU cv2.resize(resize_norm_img) 换成 csrc GPU
  kernel(ocr_rec_resize_norm)，批内单次 stack → self.net(已是 CRNN TRT) → postprocess。
  尺寸公式 1:1 复刻原 resize_norm_img。CRNN net.forward 已被 pipeline 注入为 TRT。
- FastTextDetector：覆写 _preprocess_det_image(CPU DetResizeForTest → GPU ocr_preprocess_image
  kernel) 与 _batch_process_preprocessed(np.stack+H2D → torch.stack GPU 常驻)。DBNet net.forward
  已被 pipeline 注入为 TRT。_build_det_preds/_postprocess_det_batch 等后处理照调父类。
  尺寸公式 1:1 复刻 DetResizeForTest.resize_image_type0(dst=round(x/32)*32)。这是原版
  fast_ops --fast-ops 唯一加速的环节(658ms → GPU)，追平原版的关键。

csrc 不可用 / 非 cuda / 非 DB 算法时安全回退父类原方法。
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..csrc import ocr_preprocess_image, ocr_rec_resize_norm

# rec 目标宽度量化步长(显存稳定)：CRNN TRT width profile 全动态 [16,2560]，任意宽合法。
# 每批 stack 目标宽向上取整到 _REC_WIDTH_QUANTUM 的倍数 → TRT 只见有限种输入宽度，caching
# allocator 的 buffer 尺寸从"几十上百种"收敛，reserved 碎片大幅收窄。pad 到量化宽(归一化后
# 0 值，CRNN 训练/CTC 解码本就鲁棒)——仅 pad 稍多，内容(resized_w)不变。
#
# 步长选择(实测同一 PDF，vs 无量化的识别文本逐块对比)：
#   步长   reserved峰值   相对无量化的差异
#   16(≈无量化)  5.12GB   0
#   96          3.00GB   2 块(一处公式乱码双方都错、一处 pad 后反而多识别出正确句号 → 实质无损)
#   几何13档     1.93GB   4 块(多出 1 处数学符号音标变体 ī/ñ + 1 处直/弯引号样式，噪声级)
# 默认 96：无损约束下的显存最优点(-41%)。调小→更贴无量化(显存回升)；调大→更省显存(可能引入
# 单字符级 CTC 边界抖动，非整词丢失)。CRNN 宽方向下采样使 96 的倍数落在整特征步上，边界更稳。
_REC_WIDTH_QUANTUM = 96

# rec 单批"批大小×量化后宽度"像素预算。CRNN 的 CTC logits 是 [B, T≈W/4, V≈6625] fp32，
# 大小 ∝ B×W：B=6 + W=2560(实测最宽档) 时单 logits 就 >1GB，postprocess 再复制一份，
# 整文档 1213 行按宽度升序扫过时 reserved 一路爬到 ~4.7GB(每文档末尾的显存尖峰)。
# 预算 5120 = 2×2560：最宽档 B 自动降到 2，logits ≤ ~350MB；窄行批次不变(仍 6)。
# 只拆不并，窄行分批与原固定 batch_num 完全一致(零精度影响)，宽行只是 pad 更少。
_REC_BW_BUDGET = 5120


def _quantize_rec_width(w: int, max_width: int = 2560) -> int:
    """把目标宽 w 向上取整到 _REC_WIDTH_QUANTUM 的倍数(不超过 max_width)。"""
    q = _REC_WIDTH_QUANTUM
    return min(((int(w) + q - 1) // q) * q, max_width)


def _adaptive_rec_spans(widths, batch_num: int):
    """按(已排序)宽度把 [0, n) 切成 (beg, end) 段：段内 数量×段最大宽 ≤ _REC_BW_BUDGET，
    且数量 ≤ batch_num。widths 为按升序排列的量化后目标宽。只拆不并。"""
    n = len(widths)
    spans = []
    beg = 0
    while beg < n:
        end = beg + 1
        while end < n and end - beg < batch_num \
                and (end - beg + 1) * widths[end] <= _REC_BW_BUDGET:
            end += 1
        spans.append((beg, end))
        beg = end
    return spans


def _gpu_resize_norm_batch(recognizer, img_list, indices, beg, end):
    """对 img_list[indices[beg:end]] 做 GPU resize_norm，返回 [N,3,48,imgW] fp32 GPU tensor。

    尺寸公式复刻 TextRecognizer.resize_norm_img(predict_rec.py)：
      imgC,imgH,imgW0 = rec_image_shape; max_wh_ratio=max(批内, imgW0/imgH)
      imgW = clamp(int(imgH*max_wh_ratio), limited_min_width, limited_max_width)
      每图 resized_w = min(imgW, ceil(imgH*w/h) 下限 limited_min_width)
    """
    imgC, imgH, imgW0 = recognizer.rec_image_shape
    width_list = [img_list[i].shape[1] / float(img_list[i].shape[0]) for i in range(len(img_list))]
    max_wh_ratio = imgW0 / imgH
    for k in range(beg, end):
        max_wh_ratio = max(max_wh_ratio, width_list[indices[k]])
    imgW = int(imgH * max_wh_ratio)
    imgW = max(min(imgW, recognizer.limited_max_width), recognizer.limited_min_width)
    # 量化 pad 目标宽到固定档位(显存稳定)。resized_w(内容宽)不变，只多 pad 右侧。
    imgW = min(_quantize_rec_width(imgW, recognizer.limited_max_width), recognizer.limited_max_width)

    outs = []
    for k in range(beg, end):
        img = img_list[indices[k]]           # BGR uint8 [h,w,3]
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = max(math.ceil(imgH * ratio), recognizer.limited_min_width)
        resized_w = min(imgW, int(ratio_imgH))
        resized_w = max(1, min(resized_w, imgW))
        src = torch.from_numpy(np.ascontiguousarray(img)).cuda()  # 一次 H2D
        dst = ocr_rec_resize_norm(src, resized_w, imgW)           # [3,48,imgW] fp32 GPU
        outs.append(dst)
    return torch.stack(outs, dim=0)


def _compute_det_size(h, w, limit_side_len=960):
    """复刻 DetResizeForTest.resize_image_type0：长边限到 limit_side_len，再 round 到 32 的倍数。"""
    if max(h, w) > limit_side_len:
        ratio = limit_side_len / max(h, w) if h != w else limit_side_len / h
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
    else:
        resize_h, resize_w = h, w
    dst_h = max(int(round(resize_h / 32) * 32), 32)
    dst_w = max(int(round(resize_w / 32) * 32), 32)
    return dst_h, dst_w


class FastTextDetector:
    """混入类：覆写 det 预处理为 GPU csrc kernel，批内 torch.stack GPU 常驻。

    仅覆写两个自有方法，后处理/过滤照调父类 → 对 MinerU 升级免疫。仅 DB 系算法启用，
    其余(EAST/SAST/FCE/PSE)回退父类 CPU 路径(语义不符 kernel)。
    """

    def _preprocess_det_image(self, img):
        """CPU numpy/GPU tensor BGR uint8 → (GPU float32 [3,dst_h,dst_w], shape_list, ori_shape)。

        尺寸/归一化 1:1 复刻 DetResizeForTest+NormalizeImage+ToCHW。非 DB 算法回退父类。
        """
        if getattr(self, "det_algorithm", "DB") not in ("DB", "DB++"):
            return super()._preprocess_det_image(img)
        try:
            h, w = img.shape[:2]
            limit_side_len = getattr(getattr(self, "args", None), "det_limit_side_len", 960)
            dst_h, dst_w = _compute_det_size(h, w, limit_side_len)
            if not isinstance(img, torch.Tensor) or img.device.type != "cuda":
                src = torch.from_numpy(np.ascontiguousarray(img)).to(
                    device=self.device, dtype=torch.uint8)
            else:
                src = img.to(dtype=torch.uint8)
            dst = ocr_preprocess_image(src, dst_h, dst_w)          # [3,dst_h,dst_w] fp32 GPU
            shape_list = np.array([[h, w, dst_h / h, dst_w / w]], dtype=np.float32)
            return dst, shape_list, (h, w)
        except Exception:
            return super()._preprocess_det_image(img)

    def _batch_process_preprocessed(self, batch_items):
        """batch_items 里 img_processed 已是 GPU tensor → torch.stack(免 np.stack+H2D)→ net(TRT)。

        后处理走父类 _build_det_preds/_postprocess_det_batch。任一 item 非 GPU tensor 时回退父类
        (与父类 np 路径兼容)。
        """
        if not batch_items:
            return [], 0
        if not all(isinstance(it[1], torch.Tensor) for it in batch_items):
            return super()._batch_process_preprocessed(batch_items)
        starttime = time.time()
        batch_data = [it[1] for it in batch_items]
        batch_shapes = np.concatenate([it[2] for it in batch_items], axis=0)
        ori_shapes = [it[3] for it in batch_items]
        try:
            batch_tensor = torch.stack(batch_data, dim=0)
            with torch.inference_mode():
                inp = self._to_inference_dtype(batch_tensor)
                outputs = self.net(inp)                # net.forward 已注入 DBNet TRT
            preds = self._build_det_preds(outputs)
            dt_boxes_batch = self._postprocess_det_batch(preds, batch_shapes, ori_shapes)
        except Exception:
            # GPU 路径失败 → 逐图回退父类原方法(重新 CPU 预处理，安全)
            return super()._batch_process_preprocessed(batch_items)
        total = time.time() - starttime
        return [(dt, total) for dt in dt_boxes_batch], total


class FastTextRecognizer:
    """混入类：覆写 __call__，rec 预处理走 GPU kernel + 批内单 stack。

    仅覆写 __call__(其余方法/属性继承自原 TextRecognizer)。通过 __class__ 重绑注入，
    不新增 __init__ 状态。仅当 DB/普通 CRNN 路径(rec_algorithm 非特殊)时启用 GPU 预处理。
    """

    def __call__(self, img_list, tqdm_enable=False, tqdm_desc="OCR-rec Predict",
                 tqdm_progress_bar=None):
        # 特殊算法(SAR/SVTR/SRN/CAN/NRTR/RFL)走原 CPU 路径，避免语义不符。
        if getattr(self, "rec_algorithm", "CRNN") not in ("CRNN", "SVTR_LCNet", "SVTR_HGNet"):
            return super().__call__(img_list, tqdm_enable=tqdm_enable,
                                    tqdm_desc=tqdm_desc, tqdm_progress_bar=tqdm_progress_bar)

        img_num = len(img_list)
        if img_num == 0:
            return [], 0.0

        # 整页 GPU 常驻路径(FastBatchAnalyze)：img_list 里是已 resize_norm 的 [3,48,imgW]
        # GPU tensor，直接直通 CRNN TRT(变宽按宽分批 pad)，不再 crop/resize_norm。
        if torch.is_tensor(img_list[0]) and img_list[0].device.type == "cuda" \
                and getattr(img_list[0], "ndim", 0) == 3:
            try:
                return self._rec_preprocessed_gpu(img_list)
            except Exception:
                pass  # 失败 → 落到下方常规路径(会因 tensor 非 numpy 再抛，交由上游兜底)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num

        # 每图的"有效量化目标宽"(与 _gpu_resize_norm_batch 的 imgW 公式一致)，
        # 供宽度预算分批：宽档自动缩小批，避免 CTC logits [B,W/4,V] 随 B×W 膨胀到 GB 级。
        imgC, imgH, imgW0 = self.rec_image_shape
        base_ratio = imgW0 / imgH
        lim_min = getattr(self, "limited_min_width", 16)
        lim_max = getattr(self, "limited_max_width", 2560)
        eff_w = [
            _quantize_rec_width(
                max(min(int(imgH * max(base_ratio, width_list[i])), lim_max), lim_min),
                lim_max)
            for i in range(img_num)
        ]
        sorted_eff_w = [eff_w[indices[k]] for k in range(img_num)]

        try:
            for beg, end in _adaptive_rec_spans(sorted_eff_w, batch_num):
                try:
                    inp = _gpu_resize_norm_batch(self, img_list, indices, beg, end)
                except Exception:
                    # GPU 预处理失败 → 该批回退父类(整体重算，安全)
                    return super().__call__(img_list, tqdm_enable=tqdm_enable,
                                            tqdm_desc=tqdm_desc, tqdm_progress_bar=tqdm_progress_bar)
                with torch.inference_mode():
                    inp = self._to_inference_dtype(inp)
                    preds = self.net(inp)                     # net.forward 已注入 CRNN TRT
                    rec_result = self.postprocess_op(preds)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg + rno]] = rec_result[rno]
        except Exception:
            return super().__call__(img_list, tqdm_enable=tqdm_enable,
                                    tqdm_desc=tqdm_desc, tqdm_progress_bar=tqdm_progress_bar)

        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if isinstance(score, float) and math.isnan(score):
                rec_res[i] = (text, 0.0)
        return rec_res, 0.0

    def _rec_preprocessed_gpu(self, img_list):
        """img_list 是已 resize_norm 的 [3,48,imgW] GPU tensor(来自 FastBatchAnalyze)。
        按宽度排序分批、批内 pad 到**量化后档位宽**(_quantize_rec_width) → net(CRNN TRT) →
        postprocess。量化让 TRT 只见 ~13 种输入宽，caching allocator buffer 尺寸收敛，显存稳定。
        pad 值 0(归一化后中灰，CRNN 训练见过，CTC 解码鲁棒)。移植自 fast_ops.patcher line 1149。"""
        n = len(img_list)
        batch_num = self.rec_batch_num
        order = sorted(range(n), key=lambda i: img_list[i].shape[-1])
        rec_res = [["", 0.0]] * n
        max_w = getattr(self, "limited_max_width", 2560)
        # 宽度预算分批(只拆不并)：宽档批缩小，CTC logits 显存有界；窄行分批不变。
        sorted_w = [_quantize_rec_width(img_list[i].shape[-1], max_w) for i in order]
        for start, stop in _adaptive_rec_spans(sorted_w, batch_num):
            idxs = order[start:stop]
            w_max = max(img_list[i].shape[-1] for i in idxs)
            # 量化 pad 目标宽到固定档位(显存稳定)：向上取整 → w_bucket >= w_max，所有图
            # pad 到同一档位，内容不裁剪。TRT 只见 ~13 种输入宽，allocator buffer 尺寸收敛。
            w_bucket = _quantize_rec_width(w_max, max_w)
            batch_list = []
            for i in idxs:
                t = img_list[i]
                w = t.shape[-1]
                if w < w_bucket:
                    padded = torch.zeros(t.shape[0], t.shape[1], w_bucket,
                                         dtype=t.dtype, device=t.device)
                    padded[:, :, :w] = t
                    t = padded
                batch_list.append(t)
            batch = torch.stack(batch_list, dim=0)
            with torch.inference_mode():
                batch = self._to_inference_dtype(batch)
                preds = self.net(batch)                # net.forward 已注入 CRNN TRT
                rec_result = self.postprocess_op(preds)
            for k, i in enumerate(idxs):
                rec_res[i] = rec_result[k]
        for i in range(n):
            text, score = rec_res[i]
            if isinstance(score, float) and math.isnan(score):
                rec_res[i] = (text, 0.0)
        return rec_res, 0.0


def inject_ocr_gpu(ocr_model) -> bool:
    """把 ocr_model.text_recognizer 的 __class__ 提升为 FastTextRecognizer 混入子类。

    幂等；返回是否成功注入。要求 csrc 可用(import 时已校验)。__class__ 重绑不改实例状态。
    """
    rec = getattr(ocr_model, "text_recognizer", None)
    if rec is None:
        return False
    if getattr(rec.__class__, "_fast_mineru_ocr", False):
        return True  # 已注入
    base = rec.__class__
    # 动态造一个 (FastTextRecognizer, 原类) 的子类，覆写 __call__ 且 super() 指向原类。
    new_cls = type(f"Fast_{base.__name__}", (FastTextRecognizer, base), {"_fast_mineru_ocr": True})
    rec._fast_mineru_orig_class = base
    rec.__class__ = new_cls
    return True


def restore_ocr_gpu(ocr_model):
    """恢复 text_recognizer 的原 __class__。"""
    rec = getattr(ocr_model, "text_recognizer", None)
    if rec is not None and hasattr(rec, "_fast_mineru_orig_class"):
        rec.__class__ = rec._fast_mineru_orig_class
        del rec._fast_mineru_orig_class


def inject_ocr_det_gpu(ocr_model) -> bool:
    """把 ocr_model.text_detector 的 __class__ 提升为 FastTextDetector 混入子类。幂等。"""
    det = getattr(ocr_model, "text_detector", None)
    if det is None:
        return False
    if getattr(det.__class__, "_fast_mineru_det", False):
        return True
    base = det.__class__
    new_cls = type(f"Fast_{base.__name__}", (FastTextDetector, base), {"_fast_mineru_det": True})
    det._fast_mineru_orig_det_class = base
    det.__class__ = new_cls
    return True


def restore_ocr_det_gpu(ocr_model):
    """恢复 text_detector 的原 __class__。"""
    det = getattr(ocr_model, "text_detector", None)
    if det is not None and hasattr(det, "_fast_mineru_orig_det_class"):
        det.__class__ = det._fast_mineru_orig_det_class
        del det._fast_mineru_orig_det_class
