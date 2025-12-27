from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass

# ----------------------------
# JSON helpers
# ----------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from deskew import determine_skew
from paddleocr import PaddleOCR
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from ultralytics import YOLO

from utils.get_dataset import get_datasets


@contextlib.contextmanager
def suppress_output(enabled: bool = True):
    """
    Suppress BOTH stdout and stderr (works for Python prints and most native libs
    that write to the process streams).
    """
    if not enabled:
        yield
        return

    devnull = open(os.devnull, "w")
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        # Also redirect Python-level sys.stdout/err
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull

        yield
    finally:
        # restore fds
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        # restore python streams
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()


def silence_paddleocr_logs():
    # Most PaddleOCR noise comes from these loggers
    for name in ("ppocr", "paddleocr", "paddle", "PaddleOCR"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        logger.handlers.clear()

    # Paddle / glog style env flags (helpful in many installs)
    os.environ.setdefault("GLOG_minloglevel", "3")  # 0=INFO,1=WARNING,2=ERROR,3=FATAL
    os.environ.setdefault("FLAGS_minlog_level", "3")
    os.environ.setdefault("FLAGS_logtostderr", "1")


def to_jsonable(obj):
    """Recursively convert common non-JSON types to JSON-safe Python types."""
    import torch

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    # ✅ pathlib
    if isinstance(obj, Path):
        return str(obj)

    # ✅ numpy scalars/arrays
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # ✅ torch tensors
    if "torch" in globals() or "torch" in locals():
        if isinstance(obj, (torch.Tensor,)):
            return to_jsonable(obj.detach().cpu().numpy())

    return obj


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class Thresholds:
    # OCR
    ocr_min_conf: float = 0.70
    baseline_min_conf: float = 0.70
    dewarp_min_baselines: int = 15
    dewarp_polyfit_min_points: int = 10

    # Detection
    table_conf: float = 0.30

    # Classification
    digital_override_conf: float = (
        0.99  # if predicted "digital" below this => treat as camera_doc
    )


@dataclass(frozen=True)
class ModelPaths:
    weights_dir: Path = Path("./weights")
    segmentation_model: Path = Path(
        "./segmentaition_data/YOLO11_PaperSeg/weights/best.pt"
    )
    doclaynet_model: Path = Path("./weights/yolov12l-doclaynet.pt")
    cls_model: Path = Path("./cls/YOLO11_cls2/weights/best.pt")


@dataclass(frozen=True)
class OutputPaths:
    out_dir: Path = Path("./out")
    debug_dir: Path = Path("./out/debug")


@dataclass(frozen=True)
class RuntimeConfig:
    qt_qpa_platform: str = "wayland;xcb"
    paddle_lang: str = "en"
    paddle_use_textline_orientation: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    thresholds: Thresholds = Thresholds()
    models: ModelPaths = ModelPaths()
    outputs: OutputPaths = OutputPaths()
    runtime: RuntimeConfig = RuntimeConfig()


# ----------------------------
# Pipeline
# ----------------------------
StageReporter = Callable[[str], None]


class Pipeline:
    def __init__(self, cfg: PipelineConfig, console: Console):
        self.cfg = cfg
        self.console = console

        os.environ["QT_QPA_PLATFORM"] = cfg.runtime.qt_qpa_platform

        # Silence PaddleOCR logs
        silence_paddleocr_logs()

        # Models
        with suppress_output():
            self.ocr = PaddleOCR(
                lang=cfg.runtime.paddle_lang,
                use_textline_orientation=cfg.runtime.paddle_use_textline_orientation,
            )
        self.segmentation = YOLO(str(cfg.models.segmentation_model))
        self.doclaynet = YOLO(str(cfg.models.doclaynet_model))
        self.cls = (str(cfg.models.cls_model))

        # Outputs
        cfg.outputs.out_dir.mkdir(parents=True, exist_ok=True)
        cfg.outputs.debug_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Geometry / transforms ----------
    @staticmethod
    def _quad_to_aabb(quad: List[List[int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _clip_bbox(
        bbox: Tuple[int, int, int, int], img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def perspective_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self.order_points(pts.astype("float32"))
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    @staticmethod
    def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        """Rotate and expand canvas to avoid cropping."""
        h, w = image.shape[:2]
        angle_rad = math.radians(angle)

        new_w = abs(np.sin(angle_rad) * h) + abs(np.cos(angle_rad) * w)
        new_h = abs(np.sin(angle_rad) * w) + abs(np.cos(angle_rad) * h)

        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        rot_mat[0, 2] += (new_w - w) / 2.0
        rot_mat[1, 2] += (new_h - h) / 2.0

        return cv2.warpAffine(
            image,
            rot_mat,
            (int(round(new_w)), int(round(new_h))),
            borderValue=background,
        )

    # ---------- Preprocess / scan ----------
    def get_document_mask(self, img: np.ndarray) -> Optional[np.ndarray]:
        results = self.segmentation(img, verbose=False)[0]
        if results.masks is None:
            return None
        mask = results.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        return cv2.resize(mask, (img.shape[1], img.shape[0]))

    @staticmethod
    def find_document_corners(mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return box.astype(int)

    def _scan_with_fallback(self, img: np.ndarray) -> np.ndarray:
        """Find and crop the document region."""
        results = self.segmentation(img, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            best_box = None
            max_area = 0
            for b in results.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)
            if best_box:
                x1, y1, x2, y2 = best_box
                return img[y1:y2, x1:x2]

        # OpenCV fallback
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)

        cnts, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return img[y : y + h, x : x + w]

        return img

    def _preprocess(
        self, img_path: Union[str, Path], report: Optional[StageReporter] = None
    ) -> Optional[np.ndarray]:
        img_path = str(img_path)

        if report:
            report("classify")

        results = self.cls.predict(img_path, verbose=False)
        pred = results[0]
        top_idx = pred.probs.top1
        top_name = pred.names[top_idx]
        conf = float(pred.probs.top1conf.item())

        if top_name == "digital" and conf < self.cfg.thresholds.digital_override_conf:
            final_class = "camera_doc"
        else:
            final_class = top_name

        if report:
            report(f"class={final_class} ({conf:.3f})")

        image = cv2.imread(img_path)
        if image is None:
            self.console.log(f"[red]Error:[/red] Could not read image: {img_path}")
            return None

        if final_class == "digital":
            # No pre-processing needed
            return image

        if report:
            report("deskew")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(gray)
        rotated = self.rotate(image, angle, (0, 0, 0))

        if report:
            report("mask/corners")

        warped = None
        mask = self.get_document_mask(rotated)
        if mask is not None:
            corners = self.find_document_corners(mask)
            if corners is not None:
                warped = self.perspective_transform(rotated, corners)

        out = warped if warped is not None else rotated

        if report:
            report("scan/crop")

        return self._scan_with_fallback(out)

    # ---------- OCR / dewarp (optional pieces kept) ----------
    # def extract_text_baselines(self, img: np.ndarray) -> List[Tuple[int, int]]:
    #     # Using PaddleOCR .ocr() API
    #     with suppress_output():
    #         result = self.ocr.ocr(img)
    #     if not result or not result[0]:
    #         return []

    #     baselines: List[Tuple[int, int]] = []
    #     for line in result[0]:
    #         if len(line) < 2:
    #             continue

    #         box = np.array(line[0])
    #         meta = line[1]

    #         # confidence extraction
    #         if isinstance(meta, (list, tuple)) and len(meta) >= 2:
    #             conf = float(meta[1])
    #         else:
    #             conf = 1.0

    #         if conf < self.cfg.thresholds.baseline_min_conf:
    #             continue

    #         bl = box[3]
    #         br = box[2]
    #         baselines.append((int((bl[0] + br[0]) / 2), int((bl[1] + br[1]) / 2)))

    #     return baselines

    def fit_baseline_curve(
        self, points: List[Tuple[int, int]], img_width: int
    ) -> Optional[np.ndarray]:
        if len(points) < self.cfg.thresholds.dewarp_polyfit_min_points:
            return None

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)

        coeffs = np.polyfit(xs, ys, deg=2)
        poly = np.poly1d(coeffs)
        x_line = np.arange(img_width, dtype=np.float32)
        curve_y = poly(x_line)
        return curve_y - float(np.mean(curve_y))

    @staticmethod
    def build_dewarp_maps(
        img_shape: Tuple[int, int, int], displacement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img_shape[:2]
        xx, yy = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        map_x = xx
        map_y = yy - displacement[xx.astype(np.int32)]
        return map_x, map_y

    # def dewarp_with_ocr(self, img: np.ndarray) -> np.ndarray:
    #     baselines = self.extract_text_baselines(img)
    #     if len(baselines) < self.cfg.thresholds.dewarp_min_baselines:
    #         return img

    #     displacement = self.fit_baseline_curve(baselines, img.shape[1])
    #     if displacement is None:
    #         return img

    #     map_x, map_y = self.build_dewarp_maps(img.shape, displacement)
    #     return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)

    # ---------- Table OCR ----------
    def _approx_word_boxes_from_line(
        self, line_quad: List[List[int]], text: str
    ) -> List[Dict[str, Any]]:
        words = [w for w in text.strip().split() if w]
        if not words:
            return []

        x_min, y_min, x_max, y_max = self._quad_to_aabb(line_quad)
        line_w = max(1, x_max - x_min)
        line_h = max(1, y_max - y_min)

        lengths = [len(w) for w in words]
        total = max(1, sum(lengths) + max(0, len(words) - 1))  # include spaces

        out: List[Dict[str, Any]] = []
        cursor = x_min
        for i, w in enumerate(words):
            w_px = int(round(line_w * (len(w) / total)))
            space_px = int(round(line_w * (1 / total))) if i < len(words) - 1 else 0

            x1 = cursor
            x2 = min(x_max, cursor + w_px)

            word_quad = [
                [x1, y_min],
                [x2, y_min],
                [x2, y_min + line_h],
                [x1, y_min + line_h],
            ]
            out.append({"word": w, "cordinate": word_quad})
            cursor = min(x_max, x2 + space_px)

        return out

    def draw_ocr_debug(
        self,
        image: np.ndarray,
        lines: List[Dict[str, Any]],
        draw_words: bool = True,
        draw_bbox: bool = True,
        draw_quad: bool = True,
    ) -> np.ndarray:
        debug_img = image.copy()

        for line in lines:
            quad = np.array(line["line_cord"], dtype=np.int32)
            bbox = line["bbox"]
            text = line["text"]

            if draw_quad:
                cv2.polylines(
                    debug_img, [quad], isClosed=True, color=(0, 255, 0), thickness=2
                )

            if draw_bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(
                    debug_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1
                )

            cv2.putText(
                debug_img,
                text,
                (bbox[0], max(0, bbox[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            if draw_words:
                for w in line["words"]:
                    word_quad = np.array(w["cordinate"], dtype=np.int32)
                    cv2.polylines(debug_img, [word_quad], True, (255, 255, 0), 1)

        return debug_img

    def ocr_table(self, table_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        OCR a table crop and return structured lines.
        Uses PaddleOCR .predict() JSON output (as in your code).
        """
        with suppress_output():
            ocr_res = self.ocr.predict(table_img)
        if not ocr_res or not ocr_res[0]:
            return []

        ocr_json = ocr_res[0].json["res"]
        rec_polys = ocr_json["rec_polys"]
        rec_texts = ocr_json["rec_texts"]
        rec_scores = ocr_json["rec_scores"]

        lines: List[Dict[str, Any]] = []
        for line_id, (poly, text, conf) in enumerate(
            zip(rec_polys, rec_texts, rec_scores)
        ):
            text = str(text).strip()
            conf = float(conf)

            if conf < self.cfg.thresholds.ocr_min_conf or not text:
                continue

            line_quad = [[int(round(p[0])), int(round(p[1]))] for p in poly]
            xs = [p[0] for p in line_quad]
            ys = [p[1] for p in line_quad]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            lines.append(
                {
                    "line_id": line_id,
                    "line_cord": line_quad,
                    "bbox": bbox,
                    "text": text,
                    "conf": conf,
                    "words": self._approx_word_boxes_from_line(line_quad, text),
                }
            )

        return lines

    def detect_tables(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect tables using DocLayNet YOLO.
        Table class index = 8 (as in your code).
        Adds pairwise distances between detected tables.
        """
        results = self.doclaynet(image, verbose=False)[0]
        if results.boxes is None:
            return []

        tables: List[Dict[str, Any]] = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if cls_id != 8 or conf < self.cfg.thresholds.table_conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            tables.append(
                {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "confidence": conf}
            )

        for idx, t in enumerate(tables):
            t["id"] = idx

        for a in tables:
            ax, ay = a["center"]
            a["distances"] = {}
            for b in tables:
                if a["id"] == b["id"]:
                    continue
                bx, by = b["center"]
                dx, dy = bx - ax, by - ay
                a["distances"][b["id"]] = {
                    "dx": dx,
                    "dy": dy,
                    "distance": float(math.hypot(dx, dy)),
                }

        return tables

    def process(
        self,
        img_path: Union[str, Path],
        *,
        debug: bool = True,
        report: Optional[StageReporter] = None,
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        img = self._preprocess(img_path, report=report)
        if img is None:
            return None, []

        if report:
            report("detect_tables")

        tables = self.detect_tables(img)
        h, w = img.shape[:2]

        for t_idx, t in enumerate(tables):
            clipped = self._clip_bbox(t["bbox"], w, h)
            if clipped is None:
                t["lines"] = []
                continue

            x1, y1, x2, y2 = clipped
            crop = img[y1:y2, x1:x2]

            if report:
                report(f"ocr_table {t_idx + 1}/{len(tables)}")

            lines = self.ocr_table(crop)
            t["lines"] = lines

            if debug and lines:
                debug_img = self.draw_ocr_debug(crop, lines)
                debug_path = (
                    self.cfg.outputs.debug_dir
                    / f"{Path(str(img_path)).name}_table_{t_idx}_ocr_debug.jpg"
                )
                cv2.imwrite(str(debug_path), debug_img)

        if report:
            report("done")

        return img, tables


# ----------------------------
# Main
# ----------------------------
def run_pipeline() -> None:
    console = Console()
    cfg = PipelineConfig()

    # Nice startup summary
    console.print("[bold]Pipeline config[/bold]")
    console.print_json(data=to_jsonable(asdict(cfg)))

    pipeline = Pipeline(cfg, console)
    images = get_datasets()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("• [cyan]{task.fields[stage]}[/cyan]"),
        TextColumn("• {task.fields[path]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    out_dir = cfg.outputs.out_dir

    with progress:
        task = progress.add_task(
            "Processing images",
            total=len(images),
            stage="init",
            path="",
        )

        for i, image_path in enumerate(images):
            image_path = str(image_path)
            progress.update(task, path=Path(image_path).name, stage="start")

            def report(stage: str) -> None:
                # one-line updating via task fields
                progress.update(task, stage=stage)

            img, tables = pipeline.process(image_path, debug=True, report=report)
            if img is None:
                progress.advance(task, 1)
                continue

            processed_path = out_dir / f"{i}_processed.jpg"
            cv2.imwrite(str(processed_path), img)

            # Save table crops (optional)
            for j, table in enumerate(tables):
                x1, y1, x2, y2 = table["bbox"]
                crop = img[y1:y2, x1:x2]
                cv2.imwrite(str(out_dir / f"{i}_table_{j}.jpg"), crop)

            doc_json = {
                "doc_id": i,
                "image_path": image_path,
                "processed_image_path": str(processed_path),
                "image_size": {"width": int(img.shape[1]), "height": int(img.shape[0])},
                "tables": tables,
            }

            json_path = out_dir / f"{i}_doc.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(to_jsonable(doc_json), f, ensure_ascii=False, indent=2)

            progress.update(task, stage=f"saved {json_path.name}")
            progress.advance(task, 1)

    console.print("\n[green]All done.[/green]")


