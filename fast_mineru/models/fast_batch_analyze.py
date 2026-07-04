"""FastBatchAnalyze —— 自己拥有 OCR 编排，OCR-det 段整页 GPU 常驻。

背景：MinerU 的 BatchAnalyze 每次调用新建实例、OCR 段内联在 ~600 行 __call__ 里、无可覆写
子方法。要把整页数据流搬上 GPU(每页 1 次 H2D 替代 N 次 per-crop H2D，crop/cvtColor/mask 上
GPU，rec 文本行裁切 GPU 直通 CRNN TRT)，只能接管 __call__。

设计原则(用户决策：自己重写这段编排，方便后续加速 + 不怕 mineru 升级)：
- **照调 mineru**：layout/MFR/table/seal 段全部原样调用 mineru 函数(run_*_inference、
  sorted_boxes/merge_det_boxes/update_det_boxes/get_adjusted_mfdetrec_res、get_res_list_from_layout_res、
  _extract_table_inline_objects、_prune_empty_ocr_text_blocks、clean_vram……)。这些是升级敏感的纯
  编排/后处理，无加速价值，全部复用。
- **自己拥有(GPU)**：仅 OCR-det 段(~原版 line 702-840) —— 整页 H2D 常驻，ocr_crop_and_bgr +
  ocr_apply_mask kernel，det 喂 FastTextDetector(GPU tensor 跳 H2D)，_gpu_get_ocr_result_list 产出
  **已 resize_norm 的 [3,48,imgW] GPU tensor** 作为 np_img。
- rec 段(收集 + 批推理)照抄 mineru：img_crop_list 里是 GPU tensor 时，FastTextRecognizer.__call__
  会识别并直通 CRNN TRT(变宽分批 pad)，无需在此重写。

移植自 D:/project/MinerU/fast_ops/fast_ops/patcher.py(去 monkey-patch 化)，含其踩过的坑修正
(rot90 阈值用 TEXT_REC_ROTATE_RATIO=1.5，非早期误用的 0.75)。
"""
from __future__ import annotations

import copy
import html
import math
from collections import defaultdict

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from mineru.backend.pipeline.batch_analyze import (
    BatchAnalyze,
    LAYOUT_BASE_BATCH_SIZE,
    MFR_BASE_BATCH_SIZE,
    OCR_DET_BASE_BATCH_SIZE,
    TABLE_Wired_Wireless_CLS_BATCH_SIZE,
)
from mineru.backend.pipeline.model_init import (
    AtomModelSingleton,
    run_layout_inference,
    run_mfr_inference,
    run_ocr_inference,
)
from mineru.backend.pipeline.model_list import AtomicModel
from mineru.utils.bbox_utils import normalize_to_int_bbox
from mineru.utils.model_utils import (
    _get_int_bbox,
    clean_vram,
    get_res_list_from_layout_res,
)
from mineru.utils.ocr_utils import (
    OcrConfidence,
    TEXT_REC_ROTATE_RATIO,
    calculate_is_angle,
    get_adjusted_mfdetrec_res,
    merge_det_boxes,
    sorted_boxes,
    update_det_boxes,
)
from mineru.utils.pdf_image_tools import get_crop_np_img

from ..csrc import ocr_apply_mask, ocr_crop_and_bgr


def gpu_crop_text_line(src: torch.Tensor, points: np.ndarray, device) -> torch.Tensor:
    """GPU 文本行裁切(透视变换)，返回 BGR uint8 [H,W,3]。匹配 cv2.warpPerspective +
    BORDER_REPLICATE。轴对齐框走快速切片路径。移植自 fast_ops.patcher.gpu_crop_text_line。"""
    import torch.nn.functional as F

    x_coords, y_coords = points[:, 0], points[:, 1]
    if len(np.unique(x_coords)) == 2 and len(np.unique(y_coords)) == 2:
        xmin, xmax = int(np.min(x_coords)), int(np.max(x_coords))
        ymin, ymax = int(np.min(y_coords)), int(np.max(y_coords))
        return src[ymin:ymax, xmin:xmax].clone()

    src_pts = points.astype(np.float32)
    crop_w = int(max(np.linalg.norm(points[0] - points[1]),
                     np.linalg.norm(points[2] - points[3])))
    crop_h = int(max(np.linalg.norm(points[0] - points[3]),
                     np.linalg.norm(points[1] - points[2])))
    if crop_w < 1 or crop_h < 1:
        return src[0:0, 0:0]

    dst_pts = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = np.linalg.inv(M).astype(np.float32)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(crop_h, device=device, dtype=torch.float32),
        torch.arange(crop_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(grid_x)
    sx = M_inv[0, 0] * grid_x + M_inv[0, 1] * grid_y + M_inv[0, 2] * ones
    sy = M_inv[1, 0] * grid_x + M_inv[1, 1] * grid_y + M_inv[1, 2] * ones
    sz = M_inv[2, 0] * grid_x + M_inv[2, 1] * grid_y + M_inv[2, 2] * ones
    sx = sx / sz
    sy = sy / sz

    H, W = src.shape[0], src.shape[1]
    sx_norm = sx / (W - 1) * 2 - 1
    sy_norm = sy / (H - 1) * 2 - 1
    grid = torch.stack([sx_norm, sy_norm], dim=-1).unsqueeze(0)

    src_t = src.permute(2, 0, 1).unsqueeze(0).float()
    warp = F.grid_sample(src_t, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)
    warp = warp.squeeze(0).permute(1, 2, 0)
    return warp.clamp(0, 255).to(dtype=torch.uint8)


def gpu_resize_norm_text_line(img: torch.Tensor, imgW: int, device) -> torch.Tensor:
    """GPU resize + /127.5-1 + pad → [3,48,imgW] float32。匹配 resize_norm_img。
    移植自 fast_ops.patcher.gpu_resize_norm_text_line。"""
    import torch.nn.functional as F

    h, w = img.shape[:2]
    ratio = w / max(float(h), 1)
    imgH = 48
    resized_w = min(imgW, int(max(math.ceil(imgH * ratio), 16)))

    img_t = img.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(img_t, (imgH, resized_w), mode="bilinear", align_corners=False)
    resized = resized / 127.5 - 1.0
    if resized_w < imgW:
        pad = torch.zeros(1, 3, imgH, imgW, device=device, dtype=torch.float32)
        pad[:, :, :, :resized_w] = resized
        resized = pad
    return resized.squeeze(0)


def _gpu_get_ocr_result_list(ocr_res, useful_list, ocr_enable, crop_bgr_gpu, lang, device):
    """GPU 版 get_ocr_result_list —— 文本行在 GPU 裁切+resize_norm，np_img 存已预处理的
    [3,48,imgW] GPU tensor(rec 段直通 CRNN TRT)。坐标转换/过滤逻辑 1:1 复刻 ocr_utils.get_ocr_result_list。
    """
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    ocr_result_list = []

    for box_ocr_res in ocr_res:
        need_ocr_rec = False
        if len(box_ocr_res) == 2:
            p1, p2, p3, p4 = box_ocr_res[0]
            text, score = box_ocr_res[1]
            if score < OcrConfidence.min_confidence:
                continue
        else:
            p1, p2, p3, p4 = box_ocr_res
            text, score = "", 1

        poly = [p1, p2, p3, p4]
        if (p3[0] - p1[0]) < OcrConfidence.min_width:
            continue
        if calculate_is_angle(poly):
            x_center = sum(pt[0] for pt in poly) / 4
            y_center = sum(pt[1] for pt in poly) / 4
            new_h = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
            new_w = p3[0] - p1[0]
            p1 = [x_center - new_w / 2, y_center - new_h / 2]
            p2 = [x_center + new_w / 2, y_center - new_h / 2]
            p3 = [x_center + new_w / 2, y_center + new_h / 2]
            p4 = [x_center - new_w / 2, y_center + new_h / 2]

        img_crop = None
        if ocr_enable:
            try:
                tmp_box = np.array([[p1[0], p1[1]], [p2[0], p2[1]],
                                    [p3[0], p3[1]], [p4[0], p4[1]]], dtype=np.float32)
                text_crop = gpu_crop_text_line(crop_bgr_gpu, tmp_box, device)
                if text_crop.shape[0] >= 2 and text_crop.shape[1] >= 2:
                    h, w = text_crop.shape[:2]
                    # 对齐 get_rotate_crop_image(ocr_utils.py:493)：阈值 TEXT_REC_ROTATE_RATIO，
                    # 方向 np.rot90 默认 k=1。(fast_ops 早期误用 0.75+k=-1 会把近方标题误旋，已修正)
                    if h > 0 and w > 0 and h * 1.0 / w >= TEXT_REC_ROTATE_RATIO:
                        text_crop = torch.rot90(text_crop, k=1, dims=[0, 1])
                    h, w = text_crop.shape[:2]
                    ratio = max(w / float(h), 0.1)
                    imgW = int(min(max(48.0 * ratio, 16), 2560))
                    img_crop = gpu_resize_norm_text_line(text_crop, imgW, device)
            except Exception:
                img_crop = None

        p1 = [p1[0] - paste_x + xmin, p1[1] - paste_y + ymin]
        p2 = [p2[0] - paste_x + xmin, p2[1] - paste_y + ymin]
        p3 = [p3[0] - paste_x + xmin, p3[1] - paste_y + ymin]
        p4 = [p4[0] - paste_x + xmin, p4[1] - paste_y + ymin]
        bbox = normalize_to_int_bbox([p1, p2, p3, p4])
        if bbox is None:
            continue

        ocr_item = {
            "label": "ocr_text",
            "bbox": bbox,
            "score": 1.0 if ocr_enable else float(round(score, 2)),
            "text": text,
        }
        if img_crop is not None:
            ocr_item["np_img"] = img_crop   # GPU [3,48,imgW] fp32，rec 段直通
            ocr_item["lang"] = lang
            ocr_item["_need_ocr_rec"] = True
        ocr_result_list.append(ocr_item)

    return ocr_result_list


class FastBatchAnalyze(BatchAnalyze):
    """覆写 __call__，仅 OCR-det 段替换为整页 GPU 常驻。其余段落原样调 mineru。"""

    def _gpu_ocr_det_segment(self, ocr_res_list_all_page, atom_model_manager):
        """整页 GPU 常驻的 OCR-det 段。替换原版 __call__ line 702-798(text_ocr_det_batch_enabled
        分支)。产出写入各页 layout_res。"""
        from collections import defaultdict

        device = self.model.device
        mask_inline = self.mask_inline_formula_for_ocr_det

        # Phase 1：遍历每页，整图 1 次 H2D，GPU crop+BGR+mask
        all_cropped_images_info = []
        for ocr_res_list_dict in ocr_res_list_all_page:
            _lang = ocr_res_list_dict["lang"]
            np_img = ocr_res_list_dict["np_img"]
            gpu_page = torch.from_numpy(
                np.ascontiguousarray(np_img).copy()
            ).to(device=device, dtype=torch.uint8)

            for res in ocr_res_list_dict["ocr_res_list"]:
                cx0, cy0, cx1, cy1 = _get_int_bbox(res)
                paste_x, paste_y = 50, 50
                useful_list = [
                    paste_x, paste_y, cx0, cy0, cx1, cy1,
                    cx1 - cx0 + paste_x * 2, cy1 - cy0 + paste_y * 2,
                ]
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict["single_page_mfdetrec_res"], useful_list
                )
                crop_bgr = ocr_crop_and_bgr(gpu_page, cx0, cy0, cx1, cy1, paste_x, paste_y)
                if mask_inline and adjusted_mfdetrec_res:
                    mask_boxes = [b["bbox"] for b in adjusted_mfdetrec_res]
                    mask_t = torch.tensor(mask_boxes, dtype=torch.int32, device=device)
                    ocr_apply_mask(crop_bgr, mask_t)
                all_cropped_images_info.append(
                    (crop_bgr, useful_list, ocr_res_list_dict, adjusted_mfdetrec_res, _lang)
                )
            del gpu_page

        # Phase 2：按语言分组，GPU BGR tensor 直喂 FastTextDetector.batch_predict
        lang_groups = defaultdict(list)
        for info in all_cropped_images_info:
            lang_groups[info[4]].append(info)

        for lang, lang_crop_list in lang_groups.items():
            if not lang_crop_list:
                continue
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR, lang=lang
            )
            batch_images = [info[0] for info in lang_crop_list]
            det_batch_size = min(len(batch_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
            batch_results = run_ocr_inference(
                ocr_model.text_detector.batch_predict,
                batch_images, det_batch_size,
                tqdm_enable=True, tqdm_desc=f"OCR-det {lang}",
            )
            for info, (dt_boxes, _) in zip(lang_crop_list, batch_results):
                crop_bgr_gpu, useful_list, ocr_res_list_dict, adjusted_mfdetrec_res, _lang = info
                if dt_boxes is not None and len(dt_boxes) > 0:
                    dt_boxes_sorted = sorted_boxes(dt_boxes)
                    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []
                    dt_boxes_final = (
                        update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                        if dt_boxes_merged and adjusted_mfdetrec_res else dt_boxes_merged
                    )
                    if dt_boxes_final:
                        ocr_res = [box.tolist() if hasattr(box, "tolist") else box
                                   for box in dt_boxes_final]
                        ocr_result_list = _gpu_get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_list_dict["ocr_enable"],
                            crop_bgr_gpu, _lang, device,
                        )
                        ocr_res_list_dict["layout_res"].extend(ocr_result_list)
        clean_vram(device, vram_threshold=8)

    def __call__(self, images_with_extra_info: list) -> list:
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []
        self.model = self.model_manager.get_model(
            lang=None, formula_enable=self.formula_enable, table_enable=self.table_enable,
        )
        atom_model_manager = AtomModelSingleton()

        pil_images = [image for image, _, _ in images_with_extra_info]
        np_images = [np.asarray(image) for image, _, _ in images_with_extra_info]

        # ── layout(照调 mineru) ──
        images_layout_res += run_layout_inference(
            self.model.layout_model.batch_predict,
            pil_images, batch_size=min(8, self.batch_ratio * LAYOUT_BASE_BATCH_SIZE),
        )
        clean_vram(self.model.device, vram_threshold=8)

        # ── MFR(照调 mineru) ──
        if self.formula_enable:
            images_mfd_res = []
            for layout_res in images_layout_res:
                page_formula_res = []
                for res in layout_res:
                    if res.get("label") in ["display_formula", "inline_formula"]:
                        res.setdefault("latex", "")
                        page_formula_res.append(res)
                images_mfd_res.append(page_formula_res)
            images_formula_list = run_mfr_inference(
                self.model.mfr_model.batch_predict,
                images_mfd_res, np_images, batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            for image_index in range(len(np_images)):
                for formula_res, formula_with_latex in zip(
                    images_mfd_res[image_index], images_formula_list[image_index]
                ):
                    formula_res["latex"] = formula_with_latex.get("latex", "")
            clean_vram(self.model.device, vram_threshold=8)
        else:
            for layout_res in images_layout_res:
                layout_res[:] = [res for res in layout_res if res.get("label") != "inline_formula"]

        # ── 构建 OCR/表格数据结构(照调 mineru) ──
        ocr_res_list_all_page = []
        table_res_list_all_page = []
        for index in range(len(np_images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]
            table_inline_objects = (
                self._extract_table_inline_objects(
                    layout_res, np_img, formula_enable=self.formula_enable
                ) if self.table_enable else {}
            )
            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )
            ocr_res_list_all_page.append({
                "ocr_res_list": ocr_res_list, "lang": _lang, "ocr_enable": ocr_enable,
                "np_img": np_img, "single_page_mfdetrec_res": single_page_mfdetrec_res,
                "layout_res": layout_res,
            })
            for table_res in table_res_list:
                def get_crop_table_img(scale):
                    bbox = normalize_to_int_bbox(
                        [float(v) / float(scale) for v in table_res["bbox"]]
                    )
                    if bbox is None:
                        return np_img[0:0, 0:0]
                    return get_crop_np_img(bbox, np_img, scale=scale)

                wireless_table_img = get_crop_table_img(scale=1)
                wired_table_img = get_crop_table_img(scale=10 / 3)
                table_page_bbox = normalize_to_int_bbox(
                    table_res.get("bbox"), image_size=np_img.shape[:2]
                ) or [0, 0, 0, 0]
                table_res_list_all_page.append({
                    "table_res": table_res, "lang": _lang,
                    "table_img": wireless_table_img, "wired_table_img": wired_table_img,
                    "table_page_bbox": table_page_bbox,
                    "table_inline_objects": table_inline_objects.get(id(table_res), []),
                })

        # ── 表格识别(照调 mineru，原样复制原版 line 519-699) ──
        if self.table_enable:
            self._table_recognize_all(table_res_list_all_page, atom_model_manager)

        # ── OCR-det(自己拥有：整页 GPU 常驻) ──
        if self.text_ocr_det_batch_enabled:
            self._gpu_ocr_det_segment(ocr_res_list_all_page, atom_model_manager)
        else:
            # 未开批处理 → 回退父类逐张 CPU 路径(通过临时实例组合较繁，直接复用父类逻辑不划算，
            # 此分支罕见；保持正确性用原生单张路径)
            self._cpu_ocr_det_segment_fallback(ocr_res_list_all_page, atom_model_manager)

        # ── OCR-rec(照调 mineru：img_crop 为 GPU tensor 时 FastTextRecognizer 直通 CRNN TRT) ──
        self._ocr_rec_segment(images_layout_res, atom_model_manager)

        # ── seal(照调 mineru) ──
        self._seal_segment(ocr_res_list_all_page, atom_model_manager)

        for ocr_res_list_dict in ocr_res_list_all_page:
            self._prune_empty_ocr_text_blocks(
                ocr_res_list_dict["layout_res"], ocr_res_list_dict["ocr_enable"],
            )
        return images_layout_res

    # ────────────────────────────────────────────────────────────────────
    #  以下 helper 逐字转录自 mineru BatchAnalyze.__call__ 的对应内联段落
    #  (与 OCR 加速无关，仅因 OCR-det 夹在 __call__ 中间而必须一并拥有)。
    #  内部只调 self._xxx / mineru 现成函数 → 低风险；升级时按标注源行号 diff。
    # ────────────────────────────────────────────────────────────────────

    def _table_recognize_all(self, table_res_list_all_page, atom_model_manager):
        """转录自原版 __call__ line 519-699(表格识别整段，去掉外层 if self.table_enable)。"""
        # 图片旋转批量处理
        table_orientation_cls_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.TableOrientationCls,
        )
        try:
            if self.table_ori_cls_batch_enabled:
                rotate_labels = table_orientation_cls_model.batch_predict(
                    table_res_list_all_page,
                    det_batch_size=self.batch_ratio * OCR_DET_BASE_BATCH_SIZE,
                    tqdm_enable=True,
                )
                if len(rotate_labels) != len(table_res_list_all_page):
                    raise ValueError("Table orientation batch prediction result count mismatch")
                for table_res, rotate_label in zip(table_res_list_all_page, rotate_labels):
                    self._apply_table_rotate_label(table_res, rotate_label)
            else:
                for table_res in table_res_list_all_page:
                    rotate_label = table_orientation_cls_model.predict(table_res["table_img"])
                    self._apply_table_rotate_label(table_res, rotate_label)
        except Exception as e:
            logger.warning(f"Table orientation classification failed: {e}, using original image")

        # 表格分类
        table_cls_model = atom_model_manager.get_atom_model(atom_model_name=AtomicModel.TableCls)
        try:
            table_cls_model.batch_predict(
                table_res_list_all_page, batch_size=TABLE_Wired_Wireless_CLS_BATCH_SIZE
            )
        except Exception as e:
            logger.warning(f"Table classification failed: {e}, using default model")

        # 表格 OCR-det
        rec_img_lang_group = defaultdict(list)
        det_ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR, det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6, enable_merge_det_boxes=False,
        )
        table_det_items = self._build_table_ocr_det_items(table_res_list_all_page)
        if self.table_ocr_det_batch_enabled:
            det_images = [it["det_image"] for it in table_det_items]
            if det_images:
                det_batch_size = max(
                    1, min(len(det_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
                )
                batch_results = run_ocr_inference(
                    det_ocr_engine.text_detector.batch_predict,
                    det_images, det_batch_size, tqdm_enable=True, tqdm_desc="Table-ocr det",
                )
                if len(batch_results) != len(table_det_items):
                    raise ValueError("Table OCR det batch result count mismatch")
                for table_det_item, (dt_boxes, _) in zip(table_det_items, batch_results):
                    self._append_table_ocr_det_result(table_det_item, dt_boxes, rec_img_lang_group)
        else:
            for table_det_item in tqdm(table_det_items, desc="Table-ocr det"):
                ocr_result = run_ocr_inference(
                    det_ocr_engine.ocr, table_det_item["det_image"], rec=False,
                )[0]
                self._append_table_ocr_det_result(table_det_item, ocr_result, rec_img_lang_group)

        # 表格 OCR-rec，按语言分批
        for _lang, rec_img_list in rec_img_lang_group.items():
            if not rec_img_list:
                continue
            ocr_engine = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR, det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6, lang=_lang, enable_merge_det_boxes=False,
            )
            cropped_img_list = [item["cropped_img"] for item in rec_img_list]
            ocr_res_list = run_ocr_inference(
                ocr_engine.ocr, cropped_img_list, det=False,
                tqdm_enable=True, tqdm_desc=f"Table-ocr rec {_lang}",
            )[0]
            for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
                ocr_text = self._normalize_table_ocr_rec_text(ocr_res[0])
                ocr_result_item = [img_dict["dt_box"], html.escape(ocr_text), ocr_res[1]]
                if table_res_list_all_page[img_dict["table_id"]].get("ocr_result"):
                    table_res_list_all_page[img_dict["table_id"]]["ocr_result"].append(ocr_result_item)
                else:
                    table_res_list_all_page[img_dict["table_id"]]["ocr_result"] = [ocr_result_item]

        # 内联对象回填
        for table_res_dict in table_res_list_all_page:
            if not self._table_supports_inline_objects(table_res_dict):
                continue
            table_inline_objects = table_res_dict.get("table_inline_objects", [])
            if not table_inline_objects:
                continue
            table_ocr_result = table_res_dict.setdefault("ocr_result", [])
            for inline_object in table_inline_objects:
                table_ocr_result.append([
                    self._bbox_to_quad(inline_object["table_token_bbox"]),
                    inline_object["content"], inline_object["score"],
                ])
            self._sort_table_ocr_result(table_ocr_result)

        # 无线表格
        wireless_table_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.WirelessTable,
        )
        wireless_table_model.batch_predict(table_res_list_all_page)

        # 有线表格
        wired_table_res_list = []
        for table_res_dict in table_res_list_all_page:
            if (
                (table_res_dict["table_res"]["cls_label"] == AtomicModel.WirelessTable
                 and table_res_dict["table_res"]["cls_score"] < 0.9)
                or table_res_dict["table_res"]["cls_label"] == AtomicModel.WiredTable
            ):
                wired_table_res_list.append(table_res_dict)
            del table_res_dict["table_res"]["cls_label"]
            del table_res_dict["table_res"]["cls_score"]
        if wired_table_res_list:
            for table_res_dict in tqdm(wired_table_res_list, desc="Table-wired Predict"):
                if not table_res_dict.get("ocr_result", None):
                    continue
                wired_table_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.WiredTable, lang=table_res_dict["lang"],
                )
                table_res_dict["table_res"]["html"] = wired_table_model.predict(
                    table_res_dict["wired_table_img"], table_res_dict["ocr_result"],
                    table_res_dict["table_res"].get("html", None),
                )

        # HTML 清理
        for table_res_dict in table_res_list_all_page:
            html_code = table_res_dict["table_res"].get("html", "") or ""
            if "<table>" in html_code and "</table>" in html_code:
                start_index = html_code.find("<table>")
                end_index = html_code.rfind("</table>") + len("</table>")
                table_res_dict["table_res"]["html"] = html_code[start_index:end_index]

    def _cpu_ocr_det_segment_fallback(self, ocr_res_list_all_page, atom_model_manager):
        """转录自原版 __call__ line 800-840(未开 text_ocr_det_batch 的逐张 CPU 路径)。
        罕见分支；用 mineru 原生 crop_img/cvtColor/get_ocr_result_list 保正确。"""
        from mineru.utils.model_utils import crop_img
        from mineru.utils.ocr_utils import get_ocr_result_list

        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
            _lang = ocr_res_list_dict["lang"]
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR, lang=_lang
            )
            for res in ocr_res_list_dict["ocr_res_list"]:
                new_image, useful_list = crop_img(
                    res, ocr_res_list_dict["np_img"], crop_paste_x=50, crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict["single_page_mfdetrec_res"], useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                det_image = self._get_masked_det_image(bgr_image, adjusted_mfdetrec_res)
                ocr_res = run_ocr_inference(
                    ocr_model.ocr, det_image, mfd_res=adjusted_mfdetrec_res, rec=False,
                )[0]
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(
                        ocr_res, useful_list, ocr_res_list_dict["ocr_enable"], bgr_image, _lang,
                    )
                    ocr_res_list_dict["layout_res"].extend(ocr_result_list)

    def _ocr_rec_segment(self, images_layout_res, atom_model_manager):
        """转录自原版 __call__ line 842-921(OCR-rec 收集+批推理+写回)。
        img_crop 为我们产出的 GPU tensor 时，FastTextRecognizer.__call__ 识别并直通 CRNN TRT。"""
        need_ocr_lists_by_lang = {}
        img_crop_lists_by_lang = {}
        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if not layout_res_item.get("_need_ocr_rec"):
                    continue
                if "np_img" in layout_res_item and "lang" in layout_res_item:
                    lang = layout_res_item["lang"]
                    if lang not in need_ocr_lists_by_lang:
                        need_ocr_lists_by_lang[lang] = []
                        img_crop_lists_by_lang[lang] = []
                    need_ocr_lists_by_lang[lang].append((layout_res, layout_res_item))
                    img_crop_lists_by_lang[lang].append(layout_res_item["np_img"])
                    layout_res_item.pop("np_img", None)
                    layout_res_item.pop("lang", None)
                    layout_res_item.pop("_need_ocr_rec", None)

        if len(img_crop_lists_by_lang) == 0:
            return

        from mineru.utils.ocr_utils import OcrConfidence as _OcrConf
        for lang, img_crop_list in img_crop_lists_by_lang.items():
            if len(img_crop_list) == 0:
                continue
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR, lang=lang
            )
            ocr_res_list = run_ocr_inference(
                ocr_model.ocr, img_crop_list, det=False, tqdm_enable=True
            )[0]
            assert len(ocr_res_list) == len(need_ocr_lists_by_lang[lang]), (
                f"ocr_res_list: {len(ocr_res_list)}, "
                f"need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}"
            )
            items_to_remove = []
            for index, (page_layout_res, layout_res_item) in enumerate(need_ocr_lists_by_lang[lang]):
                ocr_text, ocr_score = ocr_res_list[index]
                layout_res_item["text"] = ocr_text
                layout_res_item["score"] = float(f"{ocr_score:.3f}")
                should_remove = False
                if ocr_score < _OcrConf.min_confidence:
                    should_remove = True
                else:
                    b = layout_res_item["bbox"]
                    lw, lh = b[2] - b[0], b[3] - b[1]
                    if (
                        ocr_text in [
                            "（204号", "（20", "（2", "（2号", "（20号", "号", "（204",
                            "(cid:)", "(ci:)", "(cd:1)", "cd:)", "c)", "(cd:)", "c", "id:)",
                            ":)", "√:)", "√i:)", "−i:)", "−:", "i:)",
                        ] and ocr_score < 0.8 and lw < lh
                    ):
                        should_remove = True
                if should_remove:
                    items_to_remove.append((page_layout_res, layout_res_item))
            for page_layout_res, layout_res_item in items_to_remove:
                if layout_res_item in page_layout_res:
                    page_layout_res.remove(layout_res_item)

    def _seal_segment(self, ocr_res_list_all_page, atom_model_manager):
        """转录自原版 __call__ line 923-970(印章 OCR)。"""
        seal_ocr_items = []
        for ocr_res_list_dict in ocr_res_list_all_page:
            for layout_res_item in ocr_res_list_dict["layout_res"]:
                if layout_res_item.get("label") == "seal":
                    seal_ocr_items.append((ocr_res_list_dict, layout_res_item))

        seal_ocr_model = None
        for ocr_res_list_dict, layout_res_item in tqdm(seal_ocr_items, desc="Seal Predict"):
            np_img = ocr_res_list_dict["np_img"]
            image_h, image_w = np_img.shape[:2]
            layout_res_item["text"] = ""
            seal_bbox = normalize_to_int_bbox(
                layout_res_item.get("bbox"), image_size=(image_h, image_w)
            )
            if seal_bbox is None:
                continue
            x0, y0, x1, y1 = seal_bbox
            seal_crop_rgb = np_img[y0:y1, x0:x1]
            if seal_crop_rgb.size == 0:
                continue
            if seal_ocr_model is None:
                seal_ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR, lang="seal",
                )
            seal_crop_bgr = cv2.cvtColor(seal_crop_rgb, cv2.COLOR_RGB2BGR)
            seal_ocr_res = run_ocr_inference(
                seal_ocr_model.ocr, seal_crop_bgr, det=True, rec=True
            )[0]
            if not seal_ocr_res:
                continue
            seal_texts = []
            for seal_item in seal_ocr_res:
                if not seal_item or len(seal_item) != 2:
                    continue
                rec_result = seal_item[1]
                if not rec_result or len(rec_result) < 1:
                    continue
                rec_text = rec_result[0]
                if rec_text:
                    seal_texts.append(rec_text)
            layout_res_item["text"] = seal_texts
