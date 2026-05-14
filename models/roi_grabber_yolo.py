from __future__ import annotations

"""YOLOv8-based ROI proposal helper.

Provides a small adapter to run Ultralytics YOLO on a single BGR image
and return a list of `RoiProposal` objects compatible with the existing
ROI pipeline in `models.roi_grabber`.
"""

from typing import List
import tempfile
import os
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

from .roi_grabber import RoiProposal, non_max_suppression


def yolov8_detect_proposals(
    image_bgr: np.ndarray,
    yolo_weights: str = "yolov8n.pt",
    conf_thresh: float = 0.3,
    nms_iou: float = 0.5,
    roi_size: int = 128,
) -> List[RoiProposal]:
    """Run YOLOv8 on the provided BGR image and return ROI proposals.

    This helper writes a temp image file and calls Ultralytics' API to
    ensure compatibility with different ultralytics versions.
    """
    if YOLO is None:
        raise RuntimeError("ultralytics YOLO package is required for YOLO ROI proposals")

    # Save a temp image to avoid API incompatibility across versions
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    try:
        cv2.imwrite(tmp_path, image_bgr)

        model = YOLO(yolo_weights)
        results = model.predict(source=tmp_path, conf=conf_thresh, verbose=False)
        if not results:
            return []
        res = results[0]
        boxes = []
        scores = []
        for box in res.boxes:
            xyxy = box.xyxy.cpu().numpy().reshape(-1)
            conf = float(box.conf.cpu().numpy().reshape(-1)[0])
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
            boxes.append((x1, y1, x2, y2))
            scores.append(conf)

        if not boxes:
            return []

        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        keep = non_max_suppression(boxes_np, scores_np, iou_threshold=nms_iou)

        proposals: List[RoiProposal] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = float((x2 - x1) * (y2 - y1))
            proposals.append(RoiProposal(box=(x1, y1, x2, y2), centroid=(cx, cy), area=area))

        return proposals
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
