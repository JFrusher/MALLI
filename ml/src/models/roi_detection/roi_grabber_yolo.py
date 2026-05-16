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
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

from .roi_grabber import RoiProposal, non_max_suppression


@dataclass(frozen=True)
class YoloDetection:
    box: tuple[int, int, int, int]
    centroid: tuple[int, int]
    area: float
    confidence: float


def yolov8_detect_detections(
    image_bgr: np.ndarray,
    yolo_weights: str = "yolov8n.pt",
    conf_thresh: float = 0.3,
    nms_iou: float = 0.5,
    roi_size: int = 128,
) -> List[YoloDetection]:
    """Run YOLOv8 on the provided BGR image and return filtered detections.

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

        detections: List[YoloDetection] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = float((x2 - x1) * (y2 - y1))
            detections.append(
                YoloDetection(
                    box=(x1, y1, x2, y2),
                    centroid=(cx, cy),
                    area=area,
                    confidence=float(scores[idx]),
                )
            )

        return detections
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def yolov8_detect_proposals(
    image_bgr: np.ndarray,
    yolo_weights: str = "yolov8n.pt",
    conf_thresh: float = 0.3,
    nms_iou: float = 0.5,
    roi_size: int = 128,
) -> List[RoiProposal]:
    detections = yolov8_detect_detections(
        image_bgr=image_bgr,
        yolo_weights=yolo_weights,
        conf_thresh=conf_thresh,
        nms_iou=nms_iou,
        roi_size=roi_size,
    )
    return [RoiProposal(box=d.box, centroid=d.centroid, area=d.area) for d in detections]


def download_roboflow_weights(
    api_key: str,
    workspace: str,
    project: str,
    version: int | str = 1,
    target_dir: str | None = None,
) -> str:
    """Download a Roboflow YOLOv8 export and return the path to a .pt weights file.

    The function requires the `roboflow` package to be installed. It calls
    `version.download("yolov8")` and searches the downloaded folder for a
    *.pt file. Raises RuntimeError on failure.
    """
    try:
        from roboflow import Roboflow
    except Exception as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError("roboflow package is required to download weights: " + str(exc))

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)
    proj = ws.project(project)
    ver = proj.version(int(version))

    out_dir = target_dir or os.path.join(os.getcwd(), "roboflow_download")
    os.makedirs(out_dir, exist_ok=True)
    dl = ver.download("yolov8", location=out_dir)

    # Search for .pt weights under the downloaded directory
    for root, _, files in os.walk(dl):
        for fn in files:
            if fn.lower().endswith(".pt"):
                return os.path.join(root, fn)

    # Try searching the top-level target_dir too
    for root, _, files in os.walk(out_dir):
        for fn in files:
            if fn.lower().endswith(".pt"):
                return os.path.join(root, fn)

    raise RuntimeError(f"No .pt weights found in Roboflow download at {dl} / {out_dir}")
