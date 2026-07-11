#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/02/07
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 图片检测和稻飞虱分类算法
# @Software: PyCharm

import json
import logging
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Optional

# 确保项目根目录在 path 中，便于作为模块导入
_FILE = Path(__file__).resolve()
_ROOT = _FILE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from script.predict.model_detect import ModelDetector
from script.predict.model_cls import ModelCls
from script.predict.model_cls_factory import ClsModel, create_classifier
from script.predict_seg_lib import (
    lookup_insect_alg_by_pattern_key,
    lookup_insect_alg_entry,
)

class PredictSize:
    """
    基于尺寸过滤的检测+分类推理器

    处理流程:
        1. 切片/整图检测 (clip_size, overlap_size 控制)
        2. 框内「全图归一化 + Otsu」目标像素占比门限（可选，剔除过亮空框）
        3. 根据尺寸配置（size.json 宽高和/或 diag_filter_range 对角线区间）过滤检测框，判断是否为疑似目标
        4. 对疑似目标调用分类模型进行识别，其余归类为 other
    """

    def __init__(self, detect_model_path, size_config_path,
                 cls_list, cls_model_path=None, offset_rate=1.2,
                 conf_thresh=0.3, conf_merge=0.1, conf_merge_draw=0.01,
                 iou_threshold=0.3, ior_threshold=0.4,
                 device=None, augment=False, half=False, cls_pad_square=False,
                 cls_gray_binarize=False,
                 cls_to_gray=False,
                 cls_backend: str | None = None,
                 timm_model: str | None = None,
                 image_size: int | None = None,
                 insect_alg: Optional[dict[str, Any]] = None,
                 insect_alg_path: str | Path | None = None,
                 insect_alg_profile: Optional[str] = None,
                 inner_boxes_fp_threshold: int = 20,
                 bin_dark_ratio_min: float = 0.2,
                 edge_dup_diag_ratio: float | None = None,
                 edge_dup_merge_strategy: str = "larger",
                 diag_filter_range: tuple[float, float] | list[float] | None = None,
                 log_prefix: str | None = None,
                 cls_deferred: bool = False,
                 nms_iou: float | None = None,
                 max_det: int | None = None,
                 nms_agnostic: bool | None = None,
                 gray_contrast_enhance: bool = False,
                 gray_clahe_clip: float = 2.0,
                 gray_clahe_tile: int = 8,
                 gray_contrast_debug_save: bool = False):
        """
        初始化检测和分类预测器（一次实例化，重复使用）

        :param detect_model_path: 检测模型路径
        :param size_config_path:  尺寸配置文件路径 (size.json)
        :param cls_list:          需要分类的类别列表, 例如 ["hefeishi", "baibeifeishi", "huifeishi"]
        :param cls_model_path:    分类模型路径, 为 None 时跳过分类步骤，仅做检测+尺寸过滤
        :param offset_rate:       尺寸容错系数, 一般 0~2, 默认 1.2
        :param conf_thresh:       检测输出置信度阈值（无 insect_alg 命中 detect_conf 时使用）
        :param conf_merge:        merge 前置信度
        :param iou_threshold:     IOU 阈值
        :param ior_threshold:     IOR 阈值
        :param device:            设备类型 ('cuda', 'mps', 'cpu')，None 则自动检测
        :param cls_pad_square:    分类前是否将非正方形裁剪白边补成正方形（与训练白边正方形一致）
        :param cls_gray_binarize: 分类前是否先做灰度+CLAHE+Otsu 二值化再扩成三通道 BGR（默认关）
        :param cls_to_gray:       分类前是否在最后一步将裁剪转灰度并扩成三通道 BGR（默认关）
        :param insect_alg:        扁平门限表（通常由 ``build_alg_table_from_out(insect_alg_all.out)`` 生成）；
                优先于 ``insect_alg_path`` 文件加载
        :param insect_alg_path:   可选 JSON 路径，显式传入时从文件加载门限表（文件不存在时告警并忽略）
        :param insect_alg_profile: 可选，指定用于解析 detect_conf 的配置键（如 daofeishi）；None 时按 cls_list 与 insect 键自动解析
        :param inner_boxes_fp_threshold: 内含框误检门限。对当前未 ``filter`` 的框，若其中**严格包含**的其它未过滤框个数
                **大于**该值，则将此外框标为误检（``filter=True``）。``<=0`` 关闭。默认 ``8``（即内含框 ≥9 时过滤外框）。
        :param bin_dark_ratio_min: 检测框裁剪经**全图灰度 min-max 归一化**后再 Otsu 二值化，将**较暗簇**视为目标
                （等价二值 1）并统计其在框内占比；占比 **低于** 该值则视为无效检测（``filter=True``，不跑分类）。
                ``<=0`` 关闭。默认 ``0.2``。
        :param edge_dup_diag_ratio: 边距-对角线比例去重。对同一 ``cls_name`` 的两框，取左/右/上/下四组对边间距，
                将其中**最小的三组距离之和**与 ``ratio * max(diag_a, diag_b)`` 比较（diag 为框对角线长）；
                若该和**小于等于**该阈值则判为重复并合并保留其一；合并时把被丢弃框与保留框做 **坐标 union**，``conf``/``cls_conf`` 取较大值。
                ``None`` 或 ``<=0`` 关闭（``predict`` 可覆盖）。默认关闭。
        :param edge_dup_merge_strategy: 重复时保留策略：``larger`` 保留面积更大框；``higher_conf`` 保留检测置信度 ``conf`` 更高框
                （相同时依次比较 ``cls_conf``、面积）。``predict`` 可覆盖。
        :param diag_filter_range: 外接矩形对角线像素区间 ``(min, max)``，仅保留 ``min <= diag <= max`` 的框；
                ``None`` 关闭。与 ``size_config_path`` 的宽高过滤可同时启用（须同时满足）。``predict`` 可覆盖。
        :param log_prefix: 日志前缀，便于多步骤管线排查（如 ``predict_all/detect_big/detect``）。
        :param cls_deferred: True 表示分类由外部步骤承担（如 predict_all 的 out/models.cls）。
        :param nms_iou: Ultralytics 内置 NMS IoU（区别于 merge 的 iou_threshold）；None 用 YOLO 默认
        :param max_det: 单图最大检测数；None 用 YOLO 默认
        :param nms_agnostic: 类无关 NMS；None 用 YOLO 默认
        :param gray_contrast_debug_save: 为 True 时由 predict_all 在推理入口传入输出目录并写出 CLAHE 预览图
        """
        self._log_prefix = (log_prefix or "").strip() or None
        self.gray_contrast_debug_save = bool(gray_contrast_debug_save)
        self.cls_list = cls_list
        self.offset_rate = offset_rate
        self.inner_boxes_fp_threshold = int(inner_boxes_fp_threshold)
        self.bin_dark_ratio_min = float(bin_dark_ratio_min)
        self.edge_dup_diag_ratio = (
            None if edge_dup_diag_ratio is None else float(edge_dup_diag_ratio)
        )
        _eds = str(edge_dup_merge_strategy or "larger").strip().lower()
        if _eds not in ("larger", "higher_conf"):
            logging.warning(
                "edge_dup_merge_strategy=%r 无效，已改为 larger", edge_dup_merge_strategy
            )
            _eds = "larger"
        self.edge_dup_merge_strategy = _eds
        self.diag_filter_min, self.diag_filter_max = self._parse_diag_filter_range(diag_filter_range)
        if self.diag_filter_min is not None:
            logging.info(
                "对角线尺寸过滤: diag in [%.1f, %.1f] px（与 size.json 宽高过滤可同时生效）",
                self.diag_filter_min,
                self.diag_filter_max,
            )
        self._insect_alg: Optional[dict[str, Any]] = None
        if insect_alg is not None:
            self._insect_alg = insect_alg
        elif insect_alg_path is not None:
            p = Path(insect_alg_path)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    self._insect_alg = json.load(f)
            else:
                logging.warning(f"insect_alg 配置文件不存在，忽略: {p}")

        effective_conf_thresh = self._resolve_detect_conf_thresh(
            conf_thresh, insect_alg_profile
        )
        if self._insect_alg and effective_conf_thresh != conf_thresh:
            logging.info(
                f"检测置信度阈值: insect_alg 使用 detect_conf={effective_conf_thresh:.4f} "
                f"(全局 conf_thresh={conf_thresh})"
            )

        # ---------- 计算尺寸过滤参数（只在 __init__ 中算一次） ----------
        _pfx = f"[{self._log_prefix}] " if self._log_prefix else ""
        if size_config_path is None:
            self.size_max = None
            self.size_min = None
            logging.info("%s无尺寸过滤参数", _pfx)
        else:
            self.size_min, self.size_max = self._compute_size_filter(
                size_config_path, cls_list, offset_rate
            )
            logging.info(
                f"尺寸过滤参数: size_min={self.size_min:.1f}, size_max={self.size_max:.1f}, "
                f"cls_list={cls_list}, offset_rate={offset_rate}"
            )

        # ---------- 加载检测模型 ----------
        self.detector = ModelDetector(
            model_path=detect_model_path,
            conf_thresh=effective_conf_thresh,
            conf_merge=conf_merge,
            conf_merge_draw=conf_merge_draw,
            iou_threshold=iou_threshold,
            ior_threshold=ior_threshold,
            device=device,
            augment=augment,
            half=half,
            nms_iou=nms_iou,
            max_det=max_det,
            nms_agnostic=nms_agnostic,
            gray_contrast_enhance=gray_contrast_enhance,
            gray_clahe_clip=gray_clahe_clip,
            gray_clahe_tile=gray_clahe_tile,
        )
        if nms_iou is not None:
            logging.info("%s检测 NMS iou=%.4f", _pfx, float(nms_iou))

        # ---------- 加载分类模型（可选） ----------
        if cls_model_path is not None:
            cls_cfg: dict[str, Any] = {}
            if cls_backend:
                cls_cfg["cls_backend"] = cls_backend
            if timm_model:
                cls_cfg["timm_model"] = timm_model
            if image_size is not None and int(image_size) > 0:
                cls_cfg["image_size"] = int(image_size)
            self.classifier: ClsModel | None = create_classifier(
                str(cls_model_path),
                pad_square=cls_pad_square,
                gray_binarize=cls_gray_binarize,
                to_gray=cls_to_gray,
                cfg=cls_cfg or None,
            )
        else:
            self.classifier = None
            if cls_deferred:
                logging.info(
                    "%sPredictSize 不内嵌分类模型（预期）：仅检测+尺寸过滤，"
                    "细分类由外部 out/models.cls 路由承担",
                    _pfx,
                )
            else:
                logging.info(
                    "%s未传递分类模型路径，将跳过分类步骤，仅做检测+尺寸过滤",
                    _pfx,
                )

    def _resolve_detect_conf_thresh(
        self, conf_thresh: float, insect_alg_profile: Optional[str]
    ) -> float:
        """
        从 insect_alg 门限表解析检测置信度：优先 profile 键，再 cls_list 顺序中带 detect_conf 的键，最后 insect。
        均无 detect_conf 时返回全局 conf_thresh。
        """
        alg = self._insect_alg
        if not alg:
            return float(conf_thresh)
        keys_try: list[str] = []
        if insect_alg_profile:
            keys_try.append(str(insect_alg_profile))
        for c in self.cls_list or []:
            cs = str(c).strip() if isinstance(c, str) else ""
            if cs and cs not in keys_try:
                keys_try.append(cs)
        if "insect" not in keys_try:
            keys_try.append("insect")
        for k in keys_try:
            entry = lookup_insect_alg_by_pattern_key(alg, k)
            if not entry:
                entry = lookup_insect_alg_entry(alg, k)
            if "detect_conf" in entry and entry["detect_conf"] is not None:
                return float(entry["detect_conf"])
        return float(conf_thresh)

    def _cls_top1_threshold_for_predicted_name(
        self, predicted_cls_name: str, cls_top1_conf_threshold: Optional[float]
    ) -> Optional[float]:
        """
        分类 top1 门限：若 insect_alg 中对该预测类名配置了 cls_conf，则用该值；
        否则用 predict 传入的 cls_top1_conf_threshold（可为 None 表示不门控）。
        """
        entry = lookup_insect_alg_entry(self._insect_alg, predicted_cls_name)
        if entry and "cls_conf" in entry and entry["cls_conf"] is not None:
            return float(entry["cls_conf"])
        if cls_top1_conf_threshold is not None:
            return float(cls_top1_conf_threshold)
        return None

    # ------------------------------------------------------------------ #
    #  尺寸配置解析
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_diag_filter_range(
        rng: tuple[float, float] | list[float] | None,
    ) -> tuple[float | None, float | None]:
        """解析对角线过滤区间；无效或关闭时返回 (None, None)。"""
        if rng is None:
            return None, None
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError(f"diag_filter_range 须为 (min, max)，当前: {rng!r}")
        lo, hi = float(rng[0]), float(rng[1])
        if lo <= 0 and hi <= 0:
            return None, None
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    @staticmethod
    def _compute_size_filter(size_config_path, cls_list, offset_rate):
        """
        根据分类列表和尺寸配置，计算最终的宽高过滤上下限

        逻辑:
            1. 遍历 cls_list 中每个类别，收集 width_px / height_px 的 min/max
            2. 取所有最小值中的最小 -> overall_min
            3. 取所有最大值中的最大 -> overall_max
            4. 应用容错系数: size_min = overall_min / offset_rate
                            size_max = overall_max * offset_rate

        :return: (size_min, size_max)
        """
        with open(size_config_path, "r", encoding="utf-8") as f:
            size_config = json.load(f)

        all_mins = []
        all_maxs = []

        for cls_name in cls_list:
            if cls_name not in size_config:
                logging.warning(f"尺寸配置中未找到类别: {cls_name}，跳过")
                continue
            cfg = size_config[cls_name]
            all_mins.append(cfg["width_px"]["min"])
            all_mins.append(cfg["height_px"]["min"])
            all_maxs.append(cfg["width_px"]["max"])
            all_maxs.append(cfg["height_px"]["max"])

        if not all_mins or not all_maxs:
            raise ValueError(
                f"无法从尺寸配置中获取有效的尺寸信息, cls_list={cls_list}"
            )

        size_min = min(all_mins) / offset_rate
        size_max = max(all_maxs) * offset_rate

        return size_min, size_max

    # ------------------------------------------------------------------ #
    #  尺寸过滤
    # ------------------------------------------------------------------ #
    def _filter_by_size(
        self,
        box,
        *,
        diag_filter_range: tuple[float, float] | list[float] | None = None,
    ):
        """
        判断检测框是否在尺寸范围内（宽高来自 size.json；对角线来自 diag_filter_range）。

        :param box: 检测结果字典，包含 x1, y1, x2, y2
        :param diag_filter_range: 单次 predict 覆盖构造时的对角线区间；None 用实例默认值
        :return: True -> 疑似需要分类; False -> 超出范围归类 other
        """
        w = box["x2"] - box["x1"]
        h = box["y2"] - box["y1"]

        if self.size_min is not None and self.size_max is not None:
            if w < self.size_min or h < self.size_min:
                return False
            if w > self.size_max or h > self.size_max:
                return False

        d_lo, d_hi = self._parse_diag_filter_range(diag_filter_range)
        if d_lo is None:
            d_lo, d_hi = self.diag_filter_min, self.diag_filter_max
        if d_lo is not None and d_hi is not None:
            diag = self._box_diag_len(box["x1"], box["y1"], box["x2"], box["y2"])
            if diag < d_lo or diag > d_hi:
                return False

        return True

    @staticmethod
    def _box_outer_strictly_contains_inner(outer: dict, inner: dict) -> bool:
        """外框几何上严格包含内框（四边均在内部），且内框面积严格小于外框。"""
        ox1, oy1, ox2, oy2 = float(outer["x1"]), float(outer["y1"]), float(outer["x2"]), float(outer["y2"])
        ix1, iy1, ix2, iy2 = float(inner["x1"]), float(inner["y1"]), float(inner["x2"]), float(inner["y2"])
        if ix1 < ox1 or iy1 < oy1 or ix2 > ox2 or iy2 > oy2:
            return False
        area_o = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        area_i = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if area_o <= 0 or area_i <= 0:
            return False
        if area_i >= area_o:
            return False
        return True

    @staticmethod
    def _detect_crop_bin_dark_ratio(image, x1, y1, x2, y2) -> float:
        """
        检测框在 ``image`` 上的裁剪区域：

        1. 全图转灰度，取**全图**最暗 ``gmin``、最亮 ``gmax``；
        2. 框内灰度按 ``(g - gmin) / (gmax - gmin)`` 线性拉伸到 ``[0, 255]``（与全图同一标尺）；
        3. 对拉伸后的框内图做 Otsu，得到 0/255 两类；
        4. **均值更小的那一类**视为目标（等价二值 1，通常为虫体等较暗区域），返回其在框内的像素占比。

        裁剪无效或 ``gmax≈gmin`` 时返回 ``0.0``。
        """
        if image is None or getattr(image, "size", 0) == 0:
            return 0.0
        h_img, w_img = image.shape[:2]
        xi1 = max(0, int(round(float(x1))))
        yi1 = max(0, int(round(float(y1))))
        xi2 = min(w_img, int(round(float(x2))))
        yi2 = min(h_img, int(round(float(y2))))
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        if image.ndim == 2:
            gray_full = np.asarray(image, dtype=np.uint8)
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return 0.0

        gmin = float(gray_full.min())
        gmax = float(gray_full.max())
        span = gmax - gmin
        if span <= 1e-6:
            return 0.0

        gray_crop = gray_full[yi1:yi2, xi1:xi2]
        if gray_crop.size == 0:
            return 0.0

        norm = (gray_crop.astype(np.float32) - gmin) / span
        norm_u8 = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)

        total = int(norm_u8.size)
        if total <= 0:
            return 0.0

        _t, bin255 = cv2.threshold(
            norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        m0 = (bin255 == 0).astype(np.uint8) * 255
        m1 = (bin255 == 255).astype(np.uint8) * 255
        c0 = int(cv2.countNonZero(m0))
        c1 = int(cv2.countNonZero(m1))
        if c0 == 0 and c1 == 0:
            return 0.0
        if c0 == 0:
            target1 = (bin255 == 255)
        elif c1 == 0:
            target1 = (bin255 == 0)
        else:
            mean_low = cv2.mean(norm_u8, mask=m0)[0]
            mean_high = cv2.mean(norm_u8, mask=m1)[0]
            # 较暗簇 → 目标=1（白底上虫体灰度经全图拉伸后仍偏低的一侧）
            target1 = (bin255 == 0) if mean_low <= mean_high else (bin255 == 255)

        n1 = int(np.count_nonzero(target1))
        return float(n1) / float(total)

    def _apply_inner_boxes_fp_filter(self, rows: list[dict], threshold: int) -> None:
        """若某外框内严格包含的（未过滤）子框个数 > threshold，则将该外框 ``filter=True`` 视为误检。"""
        if threshold <= 0 or not rows:
            return
        active_idx = [i for i, r in enumerate(rows) if not r.get("filter")]
        for i in active_idx:
            outer = rows[i]
            inner_cnt = 0
            for j in active_idx:
                if i == j:
                    continue
                if self._box_outer_strictly_contains_inner(outer, rows[j]):
                    inner_cnt += 1
            if inner_cnt > threshold:
                outer["filter"] = True
                outer["fp_inner_box_count"] = int(inner_cnt)

    @staticmethod
    def _box_diag_len(x1, y1, x2, y2) -> float:
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        return float(np.hypot(w, h))

    @staticmethod
    def _boxes_edge_dup_by_diag_ratio(a: dict, b: dict, ratio: float) -> bool:
        """
        两轴对齐框：左/右/上/下四组对边间距，取最小的三组之和；若该和小于等于 ratio * max(diag_a, diag_b) 则判重复。
        """
        if ratio <= 0:
            return False
        x1a, y1a, x2a, y2a = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
        x1b, y1b, x2b, y2b = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        diag = max(
            PredictSize._box_diag_len(x1a, y1a, x2a, y2a),
            PredictSize._box_diag_len(x1b, y1b, x2b, y2b),
        )
        if diag <= 1e-6:
            return False
        thr = float(ratio) * diag
        d_left = abs(x1a - x1b)
        d_right = abs(x2a - x2b)
        d_top = abs(y1a - y1b)
        d_bottom = abs(y2a - y2b)
        d_sorted = sorted((d_left, d_right, d_top, d_bottom))
        sum_smallest_three = d_sorted[0] + d_sorted[1] + d_sorted[2]
        return sum_smallest_three <= thr

    @staticmethod
    def _union_merge_kept_with_drop(keep: dict, drop: dict) -> None:
        """
        将 ``drop`` 的几何并入 ``keep``：外接矩形 union，检测/分类置信度取较大值（类别不变）。
        """
        kx1 = float(keep["x1"])
        ky1 = float(keep["y1"])
        kx2 = float(keep["x2"])
        ky2 = float(keep["y2"])
        lx1 = float(drop["x1"])
        ly1 = float(drop["y1"])
        lx2 = float(drop["x2"])
        ly2 = float(drop["y2"])
        keep["x1"] = int(round(min(kx1, lx1)))
        keep["y1"] = int(round(min(ky1, ly1)))
        keep["x2"] = int(round(max(kx2, lx2)))
        keep["y2"] = int(round(max(ky2, ly2)))
        try:
            keep["conf"] = max(float(keep.get("conf", 0.0) or 0.0), float(drop.get("conf", 0.0) or 0.0))
        except Exception:
            pass
        try:
            keep["cls_conf"] = max(
                float(keep.get("cls_conf", 0.0) or 0.0),
                float(drop.get("cls_conf", 0.0) or 0.0),
            )
        except Exception:
            pass

    def _apply_edge_distance_dup_merge(
        self,
        rows: list[dict],
        ratio: float | None,
        strategy: str,
    ) -> None:
        """
        同一 cls_name 的未过滤框：按「最小三组对边距之和 vs ratio*对角线」判重，贪心保留一个；重复时保留框与丢弃框做 **union** 并提升置信度，丢弃框 ``filter=True``，写 ``edge_dup_merged``。
        """
        if ratio is None or float(ratio) <= 0 or not rows:
            return
        ratio = float(ratio)
        strat = str(strategy or self.edge_dup_merge_strategy or "larger").strip().lower()
        if strat not in ("larger", "higher_conf"):
            strat = "larger"

        active_idx = [i for i, r in enumerate(rows) if not r.get("filter")]
        by_cls: dict[str, list[int]] = {}
        for i in active_idx:
            key = str(rows[i].get("cls_name", "") or "")
            by_cls.setdefault(key, []).append(i)

        def area(ii: int) -> float:
            r = rows[ii]
            return max(0.0, float(r["x2"]) - float(r["x1"])) * max(
                0.0, float(r["y2"]) - float(r["y1"])
            )

        def det_conf(ii: int) -> float:
            return float(rows[ii].get("conf", 0.0) or 0.0)

        def cls_conf(ii: int) -> float:
            return float(rows[ii].get("cls_conf", 0.0) or 0.0)

        for _cls, idxs in by_cls.items():
            if len(idxs) < 2:
                continue
            if strat == "higher_conf":
                idxs_sorted = sorted(
                    idxs,
                    key=lambda ii: (-det_conf(ii), -cls_conf(ii), -area(ii), ii),
                )
            else:
                idxs_sorted = sorted(
                    idxs,
                    key=lambda ii: (-area(ii), -det_conf(ii), -cls_conf(ii), ii),
                )
            kept: list[int] = []
            for ii in idxs_sorted:
                dup_kj: int | None = None
                for kj in kept:
                    if self._boxes_edge_dup_by_diag_ratio(rows[ii], rows[kj], ratio):
                        dup_kj = kj
                        break
                if dup_kj is not None:
                    self._union_merge_kept_with_drop(rows[dup_kj], rows[ii])
                    rows[ii]["filter"] = True
                    rows[ii]["edge_dup_merged"] = True
                else:
                    kept.append(ii)

    # ------------------------------------------------------------------ #
    #  分片检测判定（与 model_detect.ModelDetector.predict 入口一致）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _uses_clip_predict_path(w, h, clip_size, overlap_size):
        """
        True 表示走切片循环（含单张大图仅一块切片的情况），作图时可标「框到整图边缘」像素。
        与 script/predict/model_detect.py 中 predict 的分支条件保持一致。
        """
        if not clip_size or not overlap_size and (
            clip_size >= w and clip_size >= h or clip_size <= overlap_size <= 1
        ):
            return False
        return True

    @staticmethod
    def _parse_detect_clip_origin(detect_id):
        """
        解析 model_detect 切片推理写入的 detect_id，格式为 "{clip_x1}-{clip_y1}"。
        :return: (clip_x1, clip_y1) 或解析失败为 None
        """
        if not detect_id or not isinstance(detect_id, str):
            return None
        parts = detect_id.split("-", 1)
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _min_dist_to_clip_edge(x1, y1, x2, y2, clip_x1, clip_y1, clip_size, w_img, h_img):
        """
        检测框（全图坐标）到当前切片矩形四边的距离最小值，与 model_detect._is_near_clip_edge 语义一致（全图坐标版）。
        切片右下与 get_clip 一致：clip_x2 = min(w, clip_x1+clip_size)，clip_y2 同理。
        """
        clip_x2 = min(w_img, clip_x1 + clip_size)
        clip_y2 = min(h_img, clip_y1 + clip_size)
        d_left = x1 - clip_x1
        d_right = clip_x2 - x2
        d_top = y1 - clip_y1
        d_bottom = clip_y2 - y2
        return min(d_left, d_right, d_top, d_bottom)

    # ------------------------------------------------------------------ #
    #  绘制结果
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_results(
        image,
        results,
        *,
        draw_edge_px: bool = False,
        clip_size: int | None = None,
        debug_filter_palette: bool = False,
        edge_rule_enabled: bool = False,
    ):
        """
        将检测和分类结果绘制到图片上

        :param image:         原始图像 (numpy array, BGR)
        :param results:       predict 返回的结果列表
        :param draw_edge_px:  True 时在标签区增加一行「edge-N」：框到所属切片边缘的像素距离最小值 N
        :param clip_size:     与推理时切片边长一致，用于由 detect_id 还原切片右下边界
        :param debug_filter_palette: True 时按 filter 上色：未过滤红、已过滤灰（调试用）
        :param edge_rule_enabled: True 时（启用边缘距离规则判断）将距离追加到 label 内展示
        :return: 绘制后的图像副本
        """
        img_draw = image.copy()
        h_img, w_img = image.shape[:2]

        # 颜色方案: 对外输出图与 debug 一致——有效/输出框红色，other 灰色；debug_filter 模式为已过滤灰
        color_out = (0, 0, 255)
        color_other = (180, 180, 180)
        color_drop_debug = (180, 180, 180)

        for r in results:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            cls_name = r.get("cls_name", "unknown")
            cls_conf = r.get("cls_conf", 0.0)
            det_conf = r.get("conf", 0.0)

            if debug_filter_palette:
                color = color_drop_debug if r.get("filter") else color_out
            else:
                is_other = (cls_name == "other")
                color = color_other if is_other else color_out

            # 绘制矩形框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # 标签文字：先检出置信度，再分类置信度
            label = f"{cls_name} det:{det_conf:.2f} cls:{cls_conf:.2f}"
            edge_label = None
            if draw_edge_px and clip_size:
                m = r.get("edge_min_dist")
                if m is None:
                    origin = PredictSize._parse_detect_clip_origin(r.get("detect_id", ""))
                    if origin is not None:
                        cx1, cy1 = origin
                        m = PredictSize._min_dist_to_clip_edge(
                            x1, y1, x2, y2, cx1, cy1, clip_size, w_img, h_img
                        )
                if m is not None:
                    m_i = int(round(float(m)))
                    if edge_rule_enabled:
                        label = f"{label} d:{m_i}"
                    else:
                        edge_label = f"edge-{m_i}"

            # 文字背景
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            # 第一行类别/置信度，第二行（可选）edge-到切片边缘最小像素距离
            line_specs = [(label, font_scale, thickness)]
            if edge_label:
                line_specs.append((edge_label, 0.5, 1))

            # 自下而上叠行：最底行 baseline 贴近 y1（与原单行逻辑一致）
            gap = 2
            rows = []
            baseline_y = y1 - 4
            for line, fs, thk in reversed(line_specs):
                (tw, th), bl = cv2.getTextSize(line, font, fs, thk)
                rows.append((line, fs, thk, tw, th, bl, baseline_y))
                baseline_y -= th + bl + gap

            max_tw = max(r[3] for r in rows)
            pad_x = 4
            bg_bottom = y1
            bg_top = baseline_y + gap
            bg_left = x1
            bg_right = x1 + max_tw + pad_x
            if bg_top < 0:
                dy = -bg_top
                bg_top += dy
                bg_bottom += dy
                rows = [
                    (ln, fs, thk, tw, th, bl, by + dy)
                    for (ln, fs, thk, tw, th, bl, by) in rows
                ]

            cv2.rectangle(
                img_draw, (bg_left, bg_top), (bg_right, bg_bottom), color, -1
            )
            for line, fs, thk, _tw, _th, _bl, by in rows:
                cv2.putText(
                    img_draw, line, (x1 + 2, by),
                    font, fs, (255, 255, 255), thk,
                )

        return img_draw

    # ------------------------------------------------------------------ #
    #  检测 + 分类 推理
    # ------------------------------------------------------------------ #
    def predict(self, image, clip_size=2500, overlap_size=800,
                output=None, image_name=None, debug=False, debug_clip=False,
                clip_profiles: list | None = None,
                clip_start: int = 0,
                clip_batch_size: int = 1,
                edge_reject_distance=5,
                edge_reject_conf_threshold=None,
                edge_reject_cls_conf_threshold=None,
                cls_top1_conf_threshold=None,
                cls_pad_square=None,
                cls_gray_binarize=None,
                cls_to_gray=None,
                detect_gray=False,
                detect_pad_square=True,
                detect_pad_square_full_image=False,
                detect_nms_iou=None,
                detect_max_det=None,
                detect_nms_agnostic: bool | None = None,
                return_full_final=False,
                inner_boxes_fp_threshold: int | None = None,
                bin_dark_ratio_min: float | None = None,
                edge_dup_diag_ratio: float | None = None,
                edge_dup_merge_strategy: str | None = None,
                diag_filter_range: tuple[float, float] | list[float] | None = None):
        """
        完成检测和分类推理

        流程:
            1. 切片/整图 -> 调用 detect 模型
            1.5. 对每个检测框：全图 min-max 归一化后 Otsu，统计**较暗簇（目标≈1）**占比，低于门限则标为无效（``filter``）
            2. 对每个检测框按尺寸判断是否为疑似目标
            3. 疑似目标裁剪后送入分类模型
            4. 其余归类为 other
            5. 内含框误检过滤后：可选「四组对边距中最小三者之和」与 ``ratio * 对角线`` 比较，对同 cls_name 框去重合并
            6. 可选 merge 后切片边缘距离联合置信度过滤
            7. 如果指定 output 目录，将结果绘制到图片并保存

        :param image:        输入图像 (numpy array, BGR)
        :param clip_size:    切片大小, >= 全图时不切片
        :param overlap_size: 切片重叠大小 (平移算法)
        :param inner_boxes_fp_threshold: 内含框误检门限；``None`` 使用构造 ``PredictSize`` 时的默认值；``<=0`` 关闭本步
        :param bin_dark_ratio_min: 全图归一化 + Otsu 后**目标(较暗)簇**像素占比下限；``None`` 使用构造时的 ``self.bin_dark_ratio_min``；``<=0`` 关闭本步
        :param edge_dup_diag_ratio: 边距去重：四组对边距中最小三者之和 **≤** ``ratio * max(diag)`` 则判重复；合并时对保留框与丢弃框做 union；``None`` 用构造默认值；``<=0`` 关闭
        :param edge_dup_merge_strategy: 去重保留策略 ``larger`` / ``higher_conf``；``None`` 使用构造时默认值
        :param diag_filter_range: 对角线像素区间 ``(min, max)``；``None`` 使用构造时默认值；``(0,0)`` 或 ``None`` 构造值表示关闭
        :param edge_reject_distance: merge 之后到切片边缘距离阈值（像素），<=0 表示不按距离滤除
        :param edge_reject_conf_threshold: 与距离联合过滤的检测置信度阈值（不含），None 时使用检测器 conf_thresh
        :param edge_reject_cls_conf_threshold: 与距离联合过滤的分类置信度阈值（不含），None 时默认 0.0（等价不因分类置信度触发）
        :param cls_top1_conf_threshold: 分类 top1 置信度门限；不为 None 时，仅当 top1 置信度 **大于** 该值才保留模型类别名，
                否则将 cls_name 置为 other（cls_conf 仍为 top1 原始值便于排查）。None 表示不做该判定。
                若构造时设置了 insect_alg 门限表且对当前预测类名配置了 cls_conf，则对该类优先使用该门限。
        :param cls_pad_square: 本次推理是否对分类裁剪做白边正方形 padding；None 时用构造 PredictSize 时的默认值
        :param cls_gray_binarize: 本次是否对分类裁剪做灰度+CLAHE+Otsu；None 时用构造时的默认值
        :param cls_to_gray: 本次是否在分类前最后一步将裁剪转灰度（三通道 BGR）；None 时用构造时的默认值
        :param detect_gray: 是否将送入 detect 的输入图转为灰度（再转回 3 通道 BGR 以兼容 YOLO 输入）；默认 False
        :param detect_pad_square: 检测切片不足 clip_size 时是否补成正方形（原图居中、黑边）；默认 True
        :param detect_pad_square_full_image: 整图检测（不切片）时是否也先补成正方形再检测；默认 False（保持历史行为）
        :param detect_nms_iou: Ultralytics 内置 NMS 的 IoU 阈值（区别于 merge_iou 的 iou_threshold）；None 表示用 detector 默认值
        :param detect_max_det: Ultralytics 内置 NMS 的 max_det（每张图最多保留框数）；None 表示用 detector 默认值
        :param detect_nms_agnostic: Ultralytics class-agnostic NMS（跨类别抑制）；None 表示用 detector 默认值
        :param return_full_final: 为 True 时返回含 filter 标记在内的全部 final_results；默认 False 仅返回未过滤框
        :param output:       保存目录路径, 为 None 则不保存绘制结果
        :param image_name:   保存的文件名 (如 "test.jpg"), 为 None 时使用默认名 "result.jpg"
        :param debug:        调试模式
        :return: 结果列表, 每个元素为 dict:
                 {x1, y1, x2, y2, conf, cls_id, class_name, detect_id,
                  cls_name, cls_conf}；启用本门限时含 ``bin_dark_ratio``（目标簇像素占比，语义见 ``_detect_crop_bin_dark_ratio``）
        """
        img_detect = image
        if bool(detect_gray) and image is not None and getattr(image, "size", 0) != 0:
            try:
                if image.ndim == 3 and image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img_detect = cv2.merge([gray, gray, gray])
                elif image.ndim == 2:
                    img_detect = cv2.merge([image, image, image])
            except Exception:
                img_detect = image

        # ---- 第 1 步: 检测 ----
        detections = self.detector.predict(
            img_detect,
            clip_size=clip_size,
            overlap_size=overlap_size,
            clip_profiles=clip_profiles,
            clip_start=clip_start,
            clip_batch_size=clip_batch_size,
            padding=bool(detect_pad_square),
            pad_full_image_to_square=bool(detect_pad_square_full_image),
            nms_iou=detect_nms_iou,
            max_det=detect_max_det,
            nms_agnostic=detect_nms_agnostic,
            debug=debug, debug_clip=debug_clip,
            # 边缘过滤依赖分类置信度，因此在本层（分类后）统一处理；
            # detector 层传 0 仅用于保留 edge_min_dist 字段，不在 detector 层做丢弃。
            edge_reject_distance=0,
            edge_reject_conf_threshold=None,
            return_all_rows=return_full_final,
        )

        h_img, w_img = image.shape[:2]
        final_results = []

        _bin_dark_min = (
            self.bin_dark_ratio_min
            if bin_dark_ratio_min is None
            else float(bin_dark_ratio_min)
        )

        for det in detections:
            det = dict(det)
            det.setdefault("filter", False)
            if det.get("filter"):
                det.setdefault("cls_name", det.get("class_name", "other"))
                det.setdefault("cls_conf", float(det.get("conf", 0.0) or 0.0))
                det.setdefault("cls_top3", [])
                det.setdefault("cls_topk", [])
                final_results.append(det)
                continue

            # ---- 第 1.5 步: 全图归一化 + Otsu 目标占比过滤（检测出框后）----
            if _bin_dark_min > 0:
                dr = self._detect_crop_bin_dark_ratio(
                    image, det["x1"], det["y1"], det["x2"], det["y2"]
                )
                det["bin_dark_ratio"] = float(dr)
                if dr < _bin_dark_min:
                    det["cls_name"] = "other"
                    det["cls_conf"] = 0.0
                    det["cls_top3"] = []
                    det["cls_topk"] = []
                    det["filter"] = True
                    final_results.append(det)
                    continue

            # ---- 第 2 步: 尺寸过滤 ----
            if self._filter_by_size(det, diag_filter_range=diag_filter_range):
                if self.classifier is not None:
                    # 裁剪检测框区域（确保坐标不越界）
                    x1 = max(0, det["x1"])
                    y1 = max(0, det["y1"])
                    x2 = min(w_img, det["x2"])
                    y2 = min(h_img, det["y2"])

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        det["cls_name"] = "other"
                        det["cls_conf"] = 0.0
                        det["cls_top3"] = []
                        det["cls_topk"] = []
                        final_results.append(det)
                        continue

                    # ---- 第 3 步: 分类 ----
                    dev = getattr(self.detector, "device", None)
                    cls_result = self.classifier.predict(
                        crop,
                        device=dev,
                        pad_square=cls_pad_square,
                        gray_binarize=cls_gray_binarize,
                        to_gray=cls_to_gray,
                    )
                    if cls_result is not None:
                        det["cls_name"] = cls_result["class_name"]
                        det["cls_conf"] = cls_result["conf"]
                        _tk = cls_result.get("topk") or cls_result.get("top3") or []
                        det["cls_topk"] = list(_tk)
                        det["cls_top3"] = det["cls_topk"][:3]
                        cls_thr = self._cls_top1_threshold_for_predicted_name(
                            det["cls_name"], cls_top1_conf_threshold
                        )
                        if cls_thr is not None and det["cls_conf"] <= cls_thr:
                            det["cls_name"] = "other"
                    else:
                        det["cls_name"] = "other"
                        det["cls_conf"] = 0.0
                        det["cls_top3"] = []
                        det["cls_topk"] = []
                else:
                    # 无分类模型，直接使用检测模型输出的类别
                    det["cls_name"] = det.get("class_name", "unknown")
                    det["cls_conf"] = det.get("conf", 0.0)
                    _one = [
                        {
                            "class_name": str(det["cls_name"]),
                            "conf": float(det["cls_conf"] or 0.0),
                        }
                    ]
                    det["cls_topk"] = _one
                    det["cls_top3"] = _one[:3]
            else:
                # 不在尺寸范围内 -> 标记过滤，不跑分类
                det["cls_name"] = "other"
                det["cls_conf"] = 0.0
                det["cls_top3"] = []
                det["cls_topk"] = []
                det["filter"] = True

            final_results.append(det)

        # ---- 第 3.4 步: 内含框过多 -> 外框标为误检（不依赖分类/边缘）----
        _thr_ib = (
            self.inner_boxes_fp_threshold
            if inner_boxes_fp_threshold is None
            else int(inner_boxes_fp_threshold)
        )
        self._apply_inner_boxes_fp_filter(final_results, _thr_ib)

        _ed_ratio = (
            self.edge_dup_diag_ratio
            if edge_dup_diag_ratio is None
            else float(edge_dup_diag_ratio)
        )
        _ed_strat = (
            self.edge_dup_merge_strategy
            if edge_dup_merge_strategy is None
            else str(edge_dup_merge_strategy).strip().lower()
        )
        self._apply_edge_distance_dup_merge(final_results, _ed_ratio, _ed_strat)

        # ---- 第 3.5 步: merge 后边缘距离过滤（依赖分类结果）----
        if edge_reject_cls_conf_threshold is None:
            edge_reject_cls_conf_threshold = 0.0
        det_conf_threshold = (
            self.detector.conf_thresh
            if edge_reject_conf_threshold is None
            else float(edge_reject_conf_threshold)
        )
        if edge_reject_distance is not None and edge_reject_distance > 0:
            for det in final_results:
                if det.get("filter"):
                    continue
                # 分类为 other 的不处理（不做该边缘距离过滤）
                if det.get("cls_name") == "other":
                    continue
                dm = det.get("edge_min_dist", None)
                if dm is None:
                    continue
                try:
                    dm = float(dm)
                except Exception:
                    continue
                # 边缘距离大于等于阈值：有效，保留
                if dm >= float(edge_reject_distance):
                    continue
                # 边缘距离小于阈值：不直接过滤，仅当检测置信度或分类置信度低于阈值时过滤
                if det.get("conf", 0.0) < det_conf_threshold or det.get("cls_conf", 0.0) < float(edge_reject_cls_conf_threshold):
                    det["filter"] = True

        # todo 红河临时方案，灰飞虱转成白背飞虱
        for det in final_results:
            cls_name = det.get("cls_name")
            if not cls_name:
                continue

            if cls_name == "zijiaochie":
                det["cls_name"] = "zitiaochie"

        # ---- 第 4 步: 保存绘制结果 ----
        if output is not None:
            os.makedirs(output, exist_ok=True)
            save_name = image_name if image_name else "result.jpg"
            save_path = os.path.join(output, save_name)
            draw_edge = self._uses_clip_predict_path(
                w_img, h_img, clip_size, overlap_size
            )
            draw_rows = final_results if debug else [r for r in final_results if not r.get("filter")]
            img_draw = self._draw_results(
                image,
                draw_rows,
                draw_edge_px=draw_edge,
                clip_size=clip_size,
                debug_filter_palette=debug,
                    edge_rule_enabled=bool(edge_reject_distance is not None and edge_reject_distance > 0),
            )
            cv2.imwrite(save_path, img_draw)
            logging.info(f"结果图片已保存: {save_path}")

        if return_full_final:
            return final_results
        return [r for r in final_results if not r.get("filter")]

    # ------------------------------------------------------------------ #
    #  释放资源
    # ------------------------------------------------------------------ #
    def release(self):
        """释放检测模型和分类模型占用的资源"""
        if hasattr(self, "detector") and self.detector is not None:
            del self.detector.model
            self.detector = None

        if hasattr(self, "classifier") and self.classifier is not None:
            del self.classifier.model
            self.classifier = None

        logging.info("模型资源已释放")

def write_pascal_voc_xml(
    xml_path: str,
    folder_name: str,
    image_filename: str,
    width: int,
    height: int,
    depth: int,
    results,
    *,
    include_dual_conf: bool = False,
):
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    """
    将检测结果写成 Pascal VOC 格式的单个 xml 文件。
    results: predict 返回的列表，元素含 x1,y1,x2,y2, cls_name, conf 等。
    include_dual_conf: True 时在 object 下额外写入 det_conf / cls_conf。
    """
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder_name or ""
    ET.SubElement(annotation, "filename").text = image_filename
    src = ET.SubElement(annotation, "source")
    ET.SubElement(src, "database").text = "Unknown"
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text = str(int(width))
    ET.SubElement(size_el, "height").text = str(int(height))
    ET.SubElement(size_el, "depth").text = str(int(depth))
    ET.SubElement(annotation, "segmented").text = "0"

    for r in results:
        x1 = int(round(max(0, min(r["x1"], width - 1))))
        y1 = int(round(max(0, min(r["y1"], height - 1))))
        x2 = int(round(max(0, min(r["x2"], width))))
        y2 = int(round(max(0, min(r["y2"], height))))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)

        obj = ET.SubElement(annotation, "object")
        name = r.get("cls_name", r.get("class_name", "unknown"))
        ET.SubElement(obj, "name").text = str(name)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x1)
        ET.SubElement(bnd, "ymin").text = str(y1)
        ET.SubElement(bnd, "xmax").text = str(x2)
        ET.SubElement(bnd, "ymax").text = str(y2)
        if include_dual_conf:
            det_conf = float(r.get("conf", r.get("det_conf", 0.0)) or 0.0)
            cls_conf = float(r.get("cls_conf", det_conf) or det_conf)
            ET.SubElement(obj, "det_conf").text = f"{det_conf:.6f}".rstrip("0").rstrip(".")
            ET.SubElement(obj, "cls_conf").text = f"{cls_conf:.6f}".rstrip("0").rstrip(".")
        cls_topn = r.get("cls_topn")
        if isinstance(cls_topn, list) and len(cls_topn) >= 2:
            block = ET.SubElement(obj, "cls_topn")
            block.set("count", str(len(cls_topn)))
            for ent in cls_topn:
                if not isinstance(ent, dict):
                    continue
                item = ET.SubElement(block, "item")
                item.set("rank", str(int(ent.get("rank", 0) or 0)))
                ET.SubElement(item, "name").text = str(ent.get("name") or "")
                conf_v = float(ent.get("conf", 0.0) or 0.0)
                ET.SubElement(item, "conf").text = (
                    f"{conf_v:.6f}".rstrip("0").rstrip(".")
                )

    rough = ET.tostring(annotation, encoding="utf-8")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="\t", encoding="utf-8")
    os.makedirs(os.path.dirname(xml_path) or ".", exist_ok=True)
    with open(xml_path, "wb") as f:
        f.write(pretty)


current_dir = Path(__file__).parent
# ====================================================================== #
#  使用示例 — 支持给定文件夹，遍历目录及子目录下的图片
# ====================================================================== #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 支持的图片扩展名
    PIC_EXT = {".jpg", ".jpeg", ".png"}

    # daofeishi
    cls_list = ['daofeishi']

    # detect_model_path = Path( "/Users/shunyaoyin/Documents/code/models/daofeishi-detect-0405.pt")
    detect_model_path = Path("/Users/shunyaoyin/Documents/code/models/daofeishi-detect-0415.pt")
    cls_model_path = Path('/Users/shunyaoyin/Documents/code/models/daofeishi-cls.pt')
    input_path = '/Users/shunyaoyin/Documents/code/datasets/insect-data/测试集/daofeishi-0522-miss'
    # 输出目录：保存绘制结果（保持与输入相同的子目录结构和文件名）
    output_dir = input_path + "_v2"
    # v2
    augment = True  # 比赛高召回和精度场景打开，提升报出，但速度慢很多；允许一定漏报时可以关闭。
    half = True
    clip_size = 640
    overlap_size = 120
    conf_thresh = 0.6
    # 裁剪边缘框距离
    edge_reject_distance = 5
    # 边缘 检出置信度最低
    edge_reject_conf_threshold = 0.8
    # 边缘 分类置信度最低
    edge_reject_cls_conf_threshold = 0.8
    # 内含框误检：某外框内严格包含的未过滤子框数 > 该值则过滤此外框；<=0 关闭
    inner_boxes_fp_threshold = 8

    predict_debug = False
    debug_clip = False
    cls_pad_square = True
    size_config_path = None
    # 对角线尺寸过滤（像素）：仅保留 min <= sqrt(w^2+h^2) <= max；None 关闭
    diag_filter_range = (50, 260)

    t_init = time.perf_counter()
    predictor = PredictSize(
        detect_model_path=detect_model_path,
        size_config_path=size_config_path,
        cls_list=cls_list,
        cls_model_path=cls_model_path,   # 可选，设为 None 则跳过分类
        offset_rate=1.2,
        conf_thresh=conf_thresh,
        device=None,  # 自动检测
        augment=augment,
        half=half,
        diag_filter_range=diag_filter_range,
    )
    print(f"模型加载耗时 {time.perf_counter() - t_init:.2f}s")

    # ---- 收集图片列表 ----
    input_p = Path(input_path)
    if input_p.is_file():
        image_files = [input_p] if input_p.suffix.lower() in PIC_EXT else []
    elif input_p.is_dir():
        image_files = sorted(
            p for p in input_p.rglob("*") if p.is_file() and p.suffix.lower() in PIC_EXT
        )
    else:
        image_files = []
        print(f"路径不存在: {input_path}")

    print(f"共找到 {len(image_files)} 张图片")

    # ---- 逐张推理 ----
    t_batch = time.perf_counter()
    n_ok = 0
    n_skip = 0
    infer_time_sum = 0.0
    for idx, img_path in enumerate(image_files, 1):
        t_img = time.perf_counter()
        img = cv2.imread(str(img_path))
        if img is None:
            n_skip += 1
            print(f"[{idx}/{len(image_files)}] 无法读取图片，跳过: {img_path}")
            continue

        # 计算相对路径，保持输出目录结构与输入一致
        if input_p.is_dir():
            rel_path = img_path.relative_to(input_p)
        else:
            rel_path = Path(img_path.name)

        # 输出子目录 = output_dir / 相对父目录
        save_sub_dir = os.path.join(output_dir, str(rel_path.parent)) if str(rel_path.parent) != "." else output_dir

        results = predictor.predict(
            img, clip_size=clip_size, overlap_size=overlap_size,
            edge_reject_distance=edge_reject_distance,
            edge_reject_conf_threshold=edge_reject_conf_threshold,
            edge_reject_cls_conf_threshold=edge_reject_cls_conf_threshold,
            output=save_sub_dir,
            image_name=rel_path.name,
            debug=predict_debug,
            debug_clip=debug_clip,
            cls_pad_square=cls_pad_square,
            inner_boxes_fp_threshold=inner_boxes_fp_threshold,
        )

        elapsed_img = time.perf_counter() - t_img
        n_ok += 1
        infer_time_sum += elapsed_img
        print(
            f"[{idx}/{len(image_files)}] {rel_path}  检测到 {len(results)} 个目标  "
            f"耗时 {elapsed_img:.2f}s"
        )
        for r in results:
            print(
                f"    [{r['cls_name']}] conf={r.get('cls_conf', 0):.2f}  "
                f"det_conf={r['conf']:.2f}  "
                f"box=({r['x1']},{r['y1']},{r['x2']},{r['y2']})"
                f"wh={r['x2']-r['x1']},{r['y2']-r['y1']}"
            )

    # ---- 释放 ----
    predictor.release()
    total_elapsed = time.perf_counter() - t_batch
    avg_infer = infer_time_sum / n_ok if n_ok else 0.0
    print(
        f"处理完成: 成功 {n_ok} 张, 跳过 {n_skip} 张, "
        f"推理累计 {infer_time_sum:.2f}s, 整批耗时 {total_elapsed:.2f}s"
    )
    if n_ok:
        print(f"  单张平均 {avg_infer:.2f}s/张, 吞吐约 {n_ok / total_elapsed:.2f} 张/s")
