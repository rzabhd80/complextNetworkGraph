import gc
import json
import math
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np
import torch
from deskew import determine_skew
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForObjectDetection, DetrImageProcessor
from ultralytics import YOLO

from utils.get_dataset import get_datasets

os.environ["QT_QPA_PLATFORM"] = "wayland;xcb"
WIGHTS = "./weights/"


class Pipeline:
    def __init__(self):
        self.ocr = PaddleOCR(lang="en", use_textline_orientation=True)
        # logging.getLogger("ppocr").handlers = []

        # self.img = image
        self.segmentation = YOLO("./segmentaition_data/YOLO11_PaperSeg/weights/best.pt")
        # self.doclaynet = YOLO(WIGHTS + "yolov12l-doclaynet")
        self.doclaynet = YOLO(WIGHTS + "yolov12l-doclaynet.pt")
        # self.scan = YOLO()
        self.cls = YOLO("./cls/YOLO11_cls2/weights/best.pt")

    def extract_text_baselines(self, img: np.ndarray):
        result = self.ocr.ocr(img)
        if not result or not result[0]:
            return []

        baselines = []

        for line in result[0]:
            if len(line) < 2:
                continue

            box = np.array(line[0])

            meta = line[1]

            # ---- Robust confidence extraction ----
            if isinstance(meta, (list, tuple)):
                if len(meta) >= 2:
                    text = meta[0]
                    conf = float(meta[1])
                else:
                    conf = 1.0
            else:
                conf = 1.0  # no confidence provided

            if conf < 0.7:
                continue

            # Bottom edge midpoint = baseline point
            bl = box[3]
            br = box[2]

            baselines.append((int((bl[0] + br[0]) / 2), int((bl[1] + br[1]) / 2)))

        return baselines

    def fit_baseline_curve(self, points, img_width):
        if len(points) < 10:
            return None

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        coeffs = np.polyfit(xs, ys, deg=2)
        poly = np.poly1d(coeffs)

        curve_y = np.array([poly(x) for x in range(img_width)])
        mean_y = np.mean(curve_y)

        return curve_y - mean_y

    def build_dewarp_maps(self, img_shape, displacement):
        h, w = img_shape[:2]

        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        for y in range(h):
            for x in range(w):
                map_x[y, x] = x
                map_y[y, x] = y - displacement[x]

        return map_x, map_y

    def dewarp_with_ocr(self, img: np.ndarray):
        baselines = self.extract_text_baselines(img)
        if len(baselines) < 15:
            print("Not enough text for dewarp")
            return img

        displacement = self.fit_baseline_curve(baselines, img.shape[1])
        if displacement is None:
            return img

        map_x, map_y = self.build_dewarp_maps(img.shape, displacement)
        dewarped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)

        return dewarped

    def get_document_mask(self, img: np.ndarray):
        results = self.segmentation(img, verbose=False)[0]
        if results.masks is None:
            return None

        # Combine all masks (if multiple detected)
        mask = results.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        # Resize mask to image size
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        return mask

    def find_document_corners(self, mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

        # Fallback: min-area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return box.astype(int)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def perspective_transform(self, image, pts):
        rect = self.order_points(pts)
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
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def rotate(
        self,
        image: np.ndarray,
        angle: float,
        background: Union[int, Tuple[int, int, int]],
    ) -> np.ndarray:
        """Rotates an image and expands the canvas to prevent cropping corners."""
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)

        # Calculate new canvas dimensions
        width = abs(np.sin(angle_radian) * old_height) + abs(
            np.cos(angle_radian) * old_width
        )
        height = abs(np.sin(angle_radian) * old_width) + abs(
            np.cos(angle_radian) * old_height
        )

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # Adjust translation to keep image centered in new canvas
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2

        return cv2.warpAffine(
            image,
            rot_mat,
            (int(round(height)), int(round(width))),
            borderValue=background,
        )

    def _scan_with_fallback(self, img: np.ndarray):
        """Processes an image array to find and crop the document."""
        # 1. Try YOLO first (Pass the numpy array directly)
        results = self.segmentation(img, verbose=False)[0]

        if len(results.boxes) > 0:
            best_box = None
            max_area = 0
            for box_data in results.boxes:
                box = box_data.xyxy[0].cpu().numpy().astype(int)
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    best_box = box

            if best_box is not None:
                print("YOLO detected document.")
                return img[best_box[1] : best_box[3], best_box[0] : best_box[2]]

        # 2. Fallback: OpenCV Contour Detection
        print("YOLO found nothing. Using OpenCV fallback...")
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

    def _preProcess(self, imgName: str):
        results = self.cls.predict(imgName, verbose=False)
        classification = results[0]

        # Get the top prediction details
        top_class_index = classification.probs.top1
        top_class_name = classification.names[top_class_index]
        confidence = classification.probs.top1conf.item()

        # Threshold logic
        if confidence < 0.99 and top_class_name == "digital":
            final_class = "camera_doc"
        else:
            final_class = top_class_name

        print(final_class)
        if final_class == "digital":
            image = cv2.imread(imgName)
            image_ndarray = np.array(image)
            print("the image is digital no pre processing needed")
            return image_ndarray
        image = cv2.imread(imgName)
        if image is None:
            print(f"Error: Could not find {imgName}")
        else:
            print("Deskewing image...")
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(grayscale)
            rotated_img = self.rotate(image, angle, (0, 0, 0))
            mask = self.get_document_mask(rotated_img)
            if mask is not None:
                corners = self.find_document_corners(mask)
                if corners is not None:
                    warped = self.perspective_transform(rotated_img, corners)
                    # return warped

            out = warped if warped is not None else rotated_img
            # B. Detection & Cropping Phase
            print("Scanning for document...")
            final_crop = self._scan_with_fallback(out)

            # print("OCR-based dewarping...")
            # final_crop = self.dewarp_with_ocr(final_crop)

            return final_crop

    def detect_tables(self, image: np.ndarray, conf_thres: float = 0.3):
        """
        Detect tables using DocLayNet YOLO model.
        Table class index = 8
        """
        results = self.doclaynet(image, verbose=False)[0]

        if results.boxes is None:
            return []

        tables = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if cls_id == 8 and conf >= conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                tables.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

        return tables

    def process(self, imgName: str):
        # A. Pre-Processing Phase
        preProcessedImage = self._preProcess(imgName)
        tables = None
        if preProcessedImage is None:
            return None, []
        tables = self.detect_tables(preProcessedImage)

        return preProcessedImage, tables


def main():
    pipeline = Pipeline()
    images = get_datasets()

    for i, image_path in enumerate(images):
        img, tables = pipeline.process(image_path)

        if img is None:
            continue

        cv2.imwrite(f"{i}_processed.jpg", img)

        for j, table in enumerate(tables):
            x1, y1, x2, y2 = table["bbox"]
            table_crop = img[y1:y2, x1:x2]
            cv2.imwrite(f"{i}_table_{j}.jpg", table_crop)


if __name__ == "__main__":
    main()
