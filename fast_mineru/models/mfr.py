"""FastFormulaRecognizer —— MFR 预处理 GPU 化(子类化 FormulaRecognizer 覆写 batch_predict)。

CPU 路径的 crop_margin+resize+pad → 保留(PIL 内容感知裁剪，GPU 化价值低)；
normalize(222ms) + format(19ms) + ToBatch+H2D → 一个 csrc mfr_preprocess_batch kernel。
每批公式区直接上传 GPU → kernel → encoder TRT 直喂，消除逐图 numpy/PIL/cv2 转换。
"""
from __future__ import annotations

import math

import numpy as np
import torch


class FastFormulaRecognizer:
    """混入类：覆写 batch_predict，预处理(except crop_margin)走 GPU csrc kernel。"""

    def batch_predict(
        self,
        images_mfd_res: list,
        images: list,
        batch_size: int = 64,
        interline_enable: bool = True,
    ) -> list:
        import os
        from ..mineru_backend import build_mfr_batch_groups

        if not images_mfd_res:
            return []

        # ── Phase 1: 公式区裁剪(crop_margin → PIL resize → PIL pad) ——
        #                  保留 CPU，因为 crop_margin 内部 findNonZero+boundingRect 是内容感知。
        #                  PIL resize+pad 对一个 100×30 的小图是微秒级，不值得为它写 kernel。
        images_formula_list = []
        mf_image_list = []    # 384×384 numpy BGR uint8
        backfill_list = []
        image_info = []

        for mfd_res, image in zip(images_mfd_res, images):
            formula_list, crop_targets = self._build_formula_items(
                mfd_res, image, interline_enable=interline_enable,
            )
            for formula_item, (xmin, ymin, xmax, ymax) in crop_targets:
                bbox_img = image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)
                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)
                backfill_list.append(formula_item)
            images_formula_list.append(formula_list)

        if not image_info:
            return images_formula_list

        image_info.sort(key=lambda x: x[0])
        sorted_areas = [x[0] for x in image_info]
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]
        index_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)}

        # ── Phase 2: CPU UniMERNetImgDecode(crop_margin + resize + pad) ——
        #                   保留 PIL 内容感知路径。
        decoded = self.pre_tfs["UniMERNetImgDecode"](imgs=sorted_images)

        # ── Phase 3: GPU kernel(normalize + format + batch) + net ——
        #                   替代 UniMERNetTestTransform + LatexImageFormat + ToBatch + H2D。
        #                   按 build_mfr_batch_groups 逐组上传→kernel→net→释放，避免一次性
        #                   把所有公式区上传 GPU 导致 8GB VRAM 溢出。
        try:
            from ..csrc import mfr_preprocess_batch
        except Exception:
            return super().batch_predict(
                images_mfd_res, images, batch_size=batch_size,
                interline_enable=interline_enable,
            )

        formula_requested_batch_size = max(1, batch_size // 2)
        batch_groups = build_mfr_batch_groups(sorted_areas, formula_requested_batch_size)
        _amp = (
            not str(self.device).startswith("cpu")
            and os.environ.get("MFR_INFERENCE_PRECISION", "fp16").lower() != "fp32"
        )
        rec_formula = []
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=_amp):
                for batch_group in batch_groups:
                    # 只上传当前组的 decoded 图片到 GPU → kernel → net → 释放
                    group_decoded = [decoded[i] for i in batch_group]
                    try:
                        gpu_crops = [torch.from_numpy(np.ascontiguousarray(d)).cuda()
                                     for d in group_decoded]
                        inp = mfr_preprocess_batch(gpu_crops, self.device)
                    except Exception:
                        return super().batch_predict(
                            images_mfd_res, images, batch_size=batch_size,
                            interline_enable=interline_enable,
                        )
                    batch_preds = [self.net(inp)]
                    batch_preds = [p.reshape([-1]) for p in batch_preds[0]]
                    batch_preds = [bp.cpu().numpy() for bp in batch_preds]
                    rec_formula += self.post_op(batch_preds)
                    # 显式释放避免峰值叠加
                    del gpu_crops, inp, batch_preds

        unsorted_results = [""] * len(rec_formula)
        for new_idx, latex in enumerate(rec_formula):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex

        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex

        return images_formula_list


def inject_mfr_gpu(mfr_model) -> bool:
    """把 mfr_model(FormulaRecognizer)的 __class__ 提升为 FastFormulaRecognizer 混入子类。幂等。"""
    if getattr(mfr_model.__class__, "_fast_mineru_mfr", False):
        return True
    base = mfr_model.__class__
    new_cls = type(
        f"Fast_{base.__name__}", (FastFormulaRecognizer, base), {"_fast_mineru_mfr": True}
    )
    mfr_model._fast_mineru_orig_mfr_class = base
    mfr_model.__class__ = new_cls
    return True


def restore_mfr_gpu(mfr_model):
    """恢复 FormulaRecognizer 的原 __class__。"""
    if hasattr(mfr_model, "_fast_mineru_orig_mfr_class"):
        mfr_model.__class__ = mfr_model._fast_mineru_orig_mfr_class
        del mfr_model._fast_mineru_orig_mfr_class
