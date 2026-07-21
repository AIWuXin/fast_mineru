"""FastBatchAnalyze -- own OCR orchestration with per-batch inline rec to keep VRAM bounded.

Background: MinerU's BatchAnalyze creates a fresh instance each call, inlines OCR segments in a
~600-line __call__ with no overridable sub-methods. To move the whole-page data flow to GPU
(1 H2D per page instead of N per-crop H2Ds, crop/cvtColor/mask on GPU, rec text-line crop GPU
direct to CRNN TRT), we must own __call__.

Design (user decision: own the orchestration for future acceleration + mineru-upgrade safety):
- **Call mineru**: layout/MFR/table/seal via mineru functions (run_*_inference, sorted_boxes,
  merge_det_boxes, update_det_boxes, get_adjusted_mfdetrec_res, get_res_list_from_layout_res,
  _extract_table_inline_objects, _prune_empty_ocr_text_blocks, clean_vram ...). These are
  upgrade-sensitive pure orchestration/postprocessing with no accel value -- all reused.
- **Own (GPU)**: OCR-det segment (~original line 702-840) -- whole-page H2D resident,
  ocr_crop_and_bgr + ocr_apply_mask kernels, det feeds FastTextDetector (GPU tensor skips H2D),
  _gpu_get_ocr_result_list produces pre-resize_norm'd [3,48,imgW] GPU tensors as np_img.
  **Per-batch inline rec**: immediately after each det batch, run rec on the produced np_img
  GPU tensors, pop them, torch.cuda.empty_cache() -- keeps VRAM peak at one batch, not the
  whole page.

Ported from D:/project/MinerU/fast_ops/fast_ops/patcher.py (de-monkey-patched), with its
fixed rot90 threshold (TEXT_REC_ROTATE_RATIO=1.5, not the earlier mistaken 0.75).
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

# 所有 mineru 符号统一经 fast_mineru 的单一适配器边界导入（见 fast_mineru/mineru_backend.py），
# 不再直连 mineru 内部模块路径 —— 逻辑一字不改，只换 import 来源。
from ..mineru_backend import (
    BatchAnalyze,
    LAYOUT_BASE_BATCH_SIZE,
    MFR_BASE_BATCH_SIZE,
    OCR_DET_BASE_BATCH_SIZE,
    TABLE_Wired_Wireless_CLS_BATCH_SIZE,
    AtomModelSingleton,
    run_layout_inference,
    run_mfr_inference,
    run_ocr_inference,
    AtomicModel,
    normalize_to_int_bbox,
    _get_int_bbox,
    clean_vram,
    get_res_list_from_layout_res,
    OcrConfidence,
    TEXT_REC_ROTATE_RATIO,
    calculate_is_angle,
    get_adjusted_mfdetrec_res,
    merge_det_boxes,
    sorted_boxes,
    update_det_boxes,
    get_crop_np_img,
)

from ..csrc import ocr_apply_mask, ocr_crop_and_bgr

# 批内 reserved 超过此阈值才 empty_cache(见下方批循环注释)。5GB：8GB 卡上 baseline ~3.1GB
# (模型+TRT 引擎)，留 ~2GB 机动；低于阈值时 allocator 复用块，避免 cudaMalloc churn。
_EMPTY_CACHE_ABOVE_BYTES = 5 * 1024**3


# ---- GPU helpers (ported from fast_ops patcher) ------------------------------

def gpu_crop_text_line(src: torch.Tensor, points: np.ndarray, device) -> torch.Tensor:
    """GPU perspective-warp crop for text line. Returns BGR uint8 [H,W,3].
    Matches cv2.warpPerspective + BORDER_REPLICATE. Axis-aligned fast path."""
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
    """GPU resize + /127.5-1 + pad -> [3,48,imgW] float32.
    Matches resize_norm_img. Ported from fast_ops patcher."""
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
    """GPU version of get_ocr_result_list -- text-line crop + resize_norm on GPU.
    np_img stores preprocessed [3,48,imgW] GPU tensor for direct rec consumption.
    Coordinate transform / filtering 1:1 mirrors ocr_utils.get_ocr_result_list."""
    paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
    ocr_result_list = []

    for box_ocr_res in ocr_res:
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
            ocr_item["np_img"] = img_crop
            ocr_item["lang"] = lang
            ocr_item["_need_ocr_rec"] = True
        ocr_result_list.append(ocr_item)

    return ocr_result_list


# ---- FastBatchAnalyze --------------------------------------------------------

class FastBatchAnalyze(BatchAnalyze):
    """Override __call__; OCR-det with per-batch inline rec to keep VRAM bounded."""

    def _gpu_ocr_det_segment(self, ocr_res_list_all_page, atom_model_manager):
        """Whole-page GPU-resident OCR-det + per-batch inline rec.
        Crops are built lazily per det batch (not all up front), and each batch's crop
        BGR + np_img GPU tensors are consumed by det/rec then freed before the next batch
        -- VRAM peak = one batch's crops, not the whole page's. gpu_page stays resident
        (single H2D per page) until the batch loop ends."""
        from ..mineru_backend import OcrConfidence as _OcrConf

        device = self.model.device
        mask_inline = self.mask_inline_formula_for_ocr_det

        lang_set = {d["lang"] for d in ocr_res_list_all_page}
        ocr_models = {}
        for lang in lang_set:
            ocr_models[lang] = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR, lang=lang
            )

        det_batch_size = max(1, self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)

        for ocr_res_list_dict in ocr_res_list_all_page:
            _lang = ocr_res_list_dict["lang"]
            np_img_np = ocr_res_list_dict["np_img"]
            ocr_model = ocr_models[_lang]

            if not ocr_res_list_dict["ocr_res_list"]:
                continue

            # 1 H2D per page
            gpu_page = torch.from_numpy(
                np.ascontiguousarray(np_img_np).copy()
            ).to(device=device, dtype=torch.uint8)

            # 只收集元信息(CPU，零 GPU 分配)：GPU crop 推迟到批内即时建，
            # 使 VRAM 峰值 = 一个 batch 的 crop，而非整页全部 crop 常驻(8GB 卡上后者
            # 会随页面 OCR 区域数线性飙升 —— 锯齿"飙升"的支配来源)。gpu_page 保持
            # 常驻至批循环结束(单页 ~10MB，远小于所有 crop 之和)。
            page_crop_meta = []
            for res in ocr_res_list_dict["ocr_res_list"]:
                cx0, cy0, cx1, cy1 = _get_int_bbox(res)
                paste_x, paste_y = 50, 50
                useful_list = [
                    paste_x, paste_y, cx0, cy0, cx1, cy1,
                    cx1 - cx0 + paste_x * 2, cy1 - cy0 + paste_y * 2,
                ]
                adj = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict["single_page_mfdetrec_res"], useful_list
                )
                page_crop_meta.append((cx0, cy0, cx1, cy1, paste_x, paste_y, useful_list, adj))

            if not page_crop_meta:
                del gpu_page
                continue

            total = len(page_crop_meta)
            for start in range(0, total, det_batch_size):
                end = min(start + det_batch_size, total)
                meta_slice = page_crop_meta[start:end]
                # 仅为当前批即时建 crop(顺序/mask/坐标与整页预建时逐一致，仅推迟到批内)
                batch_slice = []
                for (cx0, cy0, cx1, cy1, paste_x, paste_y, useful_list, adj) in meta_slice:
                    crop_bgr = ocr_crop_and_bgr(gpu_page, cx0, cy0, cx1, cy1, paste_x, paste_y)
                    if mask_inline and adj:
                        mask_boxes = [b["bbox"] for b in adj]
                        mask_t = torch.tensor(mask_boxes, dtype=torch.int32, device=device)
                        ocr_apply_mask(crop_bgr, mask_t)
                    batch_slice.append((crop_bgr, useful_list, adj))
                batch_images = [x[0] for x in batch_slice]
                batch_results = run_ocr_inference(
                    ocr_model.text_detector.batch_predict,
                    batch_images, len(batch_slice),
                    tqdm_enable=True, tqdm_desc=f"OCR-det {_lang}",
                )

                # Collect rec candidates (GPU np_img tensors) from this batch
                batch_rec_candidates = []
                for (crop_bgr_gpu, useful_list, adj), (dt_boxes, _) \
                        in zip(batch_slice, batch_results):
                    if dt_boxes is not None and len(dt_boxes) > 0:
                        dt_sorted = sorted_boxes(dt_boxes)
                        dt_merged = merge_det_boxes(dt_sorted) if dt_sorted else []
                        dt_final = (
                            update_det_boxes(dt_merged, adj)
                            if dt_merged and adj else dt_merged
                        )
                        if dt_final:
                            ocr_res = [box.tolist() if hasattr(box, "tolist") else box
                                       for box in dt_final]
                            result = _gpu_get_ocr_result_list(
                                ocr_res, useful_list, ocr_res_list_dict["ocr_enable"],
                                crop_bgr_gpu, _lang, device,
                            )
                            for item in result:
                                if item.get("_need_ocr_rec") and "np_img" in item:
                                    batch_rec_candidates.append(item)
                                ocr_res_list_dict["layout_res"].append(item)

                # Free this batch's crop BGR tensors
                for crop_bgr, _, _ in batch_slice:
                    del crop_bgr

                # Inline rec: consume np_img GPU tensors immediately, then free them
                if batch_rec_candidates:
                    img_crops = [c["np_img"] for c in batch_rec_candidates]
                    try:
                        rec_res_list = run_ocr_inference(
                            ocr_model.ocr, img_crops, det=False, tqdm_enable=False
                        )[0]
                        items_to_remove = []
                        for idx, cand in enumerate(batch_rec_candidates):
                            ocr_text, ocr_score = rec_res_list[idx]
                            cand["text"] = ocr_text
                            cand["score"] = float(f"{ocr_score:.3f}")
                            should_remove = False
                            if ocr_score < _OcrConf.min_confidence:
                                should_remove = True
                            else:
                                b = cand["bbox"]
                                lw, lh = b[2] - b[0], b[3] - b[1]
                                if (
                                    ocr_text in [
                                        "（204号", "（20", "（2", "（2号", "（20号", "号", "（204",
                                        "(cid:)", "(ci:)", "(cd:1)", "cd:)", "c)", "(cd:)", "c", "id:)",
                                        ":)", "√:)", "√i:)", "-i:)", "-:", "i:)",
                                    ] and ocr_score < 0.8 and lw < lh
                                ):
                                    should_remove = True
                            cand.pop("np_img", None)
                            cand.pop("lang", None)
                            cand.pop("_need_ocr_rec", None)
                            if should_remove:
                                items_to_remove.append(cand)
                        for item in items_to_remove:
                            lst = ocr_res_list_dict["layout_res"]
                            if item in lst:
                                lst.remove(item)
                        del rec_res_list
                    except Exception:
                        for cand in batch_rec_candidates:
                            cand.pop("np_img", None)
                            cand.pop("lang", None)
                            cand.pop("_need_ocr_rec", None)
                    del img_crops

                del batch_rec_candidates
                del batch_images, batch_results, batch_slice, meta_slice
                # 不逐批 empty_cache：caching allocator 会复用本批释放的块(宽度量化+宽度
                # 预算分批后单批有界)，逐批 empty_cache 只会逼 allocator 反复 cudaMalloc，
                # 拖慢后续所有 stage。reserved 由页末 empty_cache + clean_vram 兜底。
                if torch.cuda.memory_reserved() > _EMPTY_CACHE_ABOVE_BYTES:
                    torch.cuda.empty_cache()

            del gpu_page, page_crop_meta
            torch.cuda.empty_cache()

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

        # -- layout (call mineru) --
        images_layout_res += run_layout_inference(
            self.model.layout_model.batch_predict,
            pil_images, batch_size=min(8, self.batch_ratio * LAYOUT_BASE_BATCH_SIZE),
        )
        clean_vram(self.model.device, vram_threshold=8)

        # -- MFR (call mineru) --
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
                layout_res[:] = [res for res in layout_res
                                 if res.get("label") != "inline_formula"]

        # -- Build OCR/table data structures (call mineru) --
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

        # -- Table recognition (call mineru) --
        if self.table_enable:
            self._table_recognize_all(table_res_list_all_page, atom_model_manager)

        # -- OCR-det (own: whole-page GPU + inline rec per batch) --
        # Rec is inlined inside _gpu_ocr_det_segment; no separate _ocr_rec_segment call.
        if self.text_ocr_det_batch_enabled:
            self._gpu_ocr_det_segment(ocr_res_list_all_page, atom_model_manager)
        else:
            self._cpu_ocr_det_segment_fallback(ocr_res_list_all_page, atom_model_manager)

        # -- Seal (call mineru) --
        self._seal_segment(ocr_res_list_all_page, atom_model_manager)

        for ocr_res_list_dict in ocr_res_list_all_page:
            self._prune_empty_ocr_text_blocks(
                ocr_res_list_dict["layout_res"], ocr_res_list_dict["ocr_enable"],
            )
        return images_layout_res

    # ---- Transcribed helpers (exact copies from mineru BatchAnalyze.__call__) ---
    # Internal calls self._xxx / mineru functions only. Low risk; diff by source line.

    def _table_recognize_all(self, table_res_list_all_page, atom_model_manager):
        """Transcribed from original __call__ line 519-699."""
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

        table_cls_model = atom_model_manager.get_atom_model(atom_model_name=AtomicModel.TableCls)
        try:
            table_cls_model.batch_predict(
                table_res_list_all_page, batch_size=TABLE_Wired_Wireless_CLS_BATCH_SIZE
            )
        except Exception as e:
            logger.warning(f"Table classification failed: {e}, using default model")

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
                    table_res_list_all_page[img_dict["table_id"]]["ocr_result"].append(
                        ocr_result_item)
                else:
                    table_res_list_all_page[img_dict["table_id"]]["ocr_result"] = [ocr_result_item]

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

        wireless_table_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.WirelessTable,
        )
        wireless_table_model.batch_predict(table_res_list_all_page)

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

        for table_res_dict in table_res_list_all_page:
            html_code = table_res_dict["table_res"].get("html", "") or ""
            if "<table>" in html_code and "</table>" in html_code:
                start_index = html_code.find("<table>")
                end_index = html_code.rfind("</table>") + len("</table>")
                table_res_dict["table_res"]["html"] = html_code[start_index:end_index]

    def _cpu_ocr_det_segment_fallback(self, ocr_res_list_all_page, atom_model_manager):
        """Transcribed from original __call__ line 800-840 (no text_ocr_det_batch path)."""
        from ..mineru_backend import crop_img, get_ocr_result_list

        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
            _lang = ocr_res_list_dict["lang"]
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR, lang=_lang
            )
            for res in ocr_res_list_dict["ocr_res_list"]:
                new_image, useful_list = crop_img(
                    res, ocr_res_list_dict["np_img"], crop_paste_x=50, crop_paste_y=50
                )
                adj = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict["single_page_mfdetrec_res"], useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                det_image = self._get_masked_det_image(bgr_image, adj)
                ocr_res = run_ocr_inference(
                    ocr_model.ocr, det_image, mfd_res=adj, rec=False,
                )[0]
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(
                        ocr_res, useful_list, ocr_res_list_dict["ocr_enable"], bgr_image, _lang,
                    )
                    ocr_res_list_dict["layout_res"].extend(ocr_result_list)

    def _seal_segment(self, ocr_res_list_all_page, atom_model_manager):
        """Transcribed from original __call__ line 923-970 (seal OCR)."""
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
