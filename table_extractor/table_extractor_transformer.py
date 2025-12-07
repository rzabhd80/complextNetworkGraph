"""
Hybrid Approach: Table Transformer + Border Extraction + Visualizations
========================================================================

CRITICAL FIXES:
1. Table coordinates NOW returned immediately after detection (Step 1)
2. OCR debugging added - saves input images and checks results
3. OCR parameters tuned for better detection
"""
import os

import torch
import numpy as np
import cv2
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForObjectDetection, DetrImageProcessor
from paddleocr import PaddleOCR
from transformers import DetrImageProcessor
from typing import List, Dict, Tuple, Optional, Any
import json
from collections import defaultdict
import gc
import warnings

warnings.filterwarnings('ignore')


class HybridTableExtractor:
    """
    Use Table Transformer for DETECTION only (finds table regions)
    Use OpenCV for border extraction (handles colored lines)
    + Visualization capabilities
    + Full debug output after each step
    """

    def __init__(self, use_fp16: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"

        # Table Transformer for DETECTION only
        self.detection_model = None
        self.processor = None

        # OCR
        self.ocr = None

        print(f"Device: {self.device}")
        if self.use_fp16:
            print("Using FP16 optimization")
        print()

    def _clear_memory(self):
        """GPU memory cleanup"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def _load_detection_model(self):
        """Load Table Transformer DETECTION model"""
        if self.detection_model is None:
            print("Loading Table Transformer (detection only)...")

            self.processor = DetrImageProcessor()

            self.detection_model = AutoModelForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection",
                revision="no_timm"
            )

            if self.use_fp16:
                self.detection_model = self.detection_model.half()

            self.detection_model = self.detection_model.to(self.device)
            self.detection_model.eval()

            if self.device == "cuda":
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"✓ Detection model loaded - VRAM: {mem:.2f} GB\n")

    def _unload_detection_model(self):
        """Unload detection model"""
        if self.detection_model is not None:
            del self.detection_model
            self.detection_model = None
            self._clear_memory()
            print("✓ Detection model unloaded\n")

    def _load_ocr(self):
        """Load OCR with improved parameters"""
        if self.ocr is None:
            print("Loading OCR with optimized settings...")

            # Enhanced OCR parameters for better detection
            # Valid parameters only - removed invalid ones
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                det_db_thresh=0.3,  # Lower threshold for text detection (default 0.3)
                det_db_box_thresh=0.5,  # Box threshold (default 0.6)
                rec_batch_num=6,  # Batch size for recognition
                # Removed: drop_score (doesn't exist)
                # Removed: use_dilation (doesn't exist)
            )
            print("✓ OCR loaded with enhanced detection parameters\n")

    def detect_tables(self, image: Image.Image, debug_prefix: str = 'debug',
                      detection_threshold: float = 0.5) -> List[Dict]:
        """
        STEP 1: Use Table Transformer to detect table REGIONS
        NOW RETURNS FULL TABLE DATA IMMEDIATELY
        """
        print("=" * 70)
        print("STEP 1: TABLE DETECTION")
        print("=" * 70)

        self._load_detection_model()

        # Prepare image
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        if self.use_fp16:
            pixel_values = pixel_values.half()

        pixel_values = pixel_values.to(self.device)

        # Run detection
        with torch.no_grad():
            outputs = self.detection_model(pixel_values)

        # Post-process with the specified threshold
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=detection_threshold
        )[0]

        # Extract table regions
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 0:  # Table class
                # Increase padding for better context in cropping
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]

                # Add a buffer
                buffer = 10
                width, height = image.size

                x1 = max(0, x1 - buffer)
                y1 = max(0, y1 - buffer)
                x2 = min(width, x2 + buffer)
                y2 = min(height, y2 + buffer)

                tables.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': score.item(),
                    'detection_method': 'table_transformer',
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                })

        # Sort by area (largest first)
        tables.sort(key=lambda t: t['area'], reverse=True)

        # SAVE DETECTION VISUALIZATION
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        debug_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_01_table_detection.png")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, table in enumerate(tables):
            x1, y1, x2, y2 = table['bbox']
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            draw.text((x1, y1 - 20), f"Table {i + 1} (conf:{table['confidence']:.2f})",
                      fill=color)
        debug_img.save(debug_path)
        print(f"✓ Saved detection visualization: {debug_path}")
        print(f"✓ Found {len(tables)} table(s) at threshold {detection_threshold}")

        if len(tables) > 0:
            print("\n📊 DETECTED TABLES WITH COORDINATES:")
            for i, table in enumerate(tables):
                print(f"  Table {i + 1}:")
                print(f"    - BBox: {table['bbox']}")
                print(f"    - Size: {table['width']}x{table['height']} pixels")
                print(f"    - Confidence: {table['confidence']:.3f}")
                print(f"    - Area: {table['area']} px²")

        # Cleanup
        del pixel_values, outputs, results
        self._clear_memory()
        self._unload_detection_model()

        print()
        return tables

    def extract_borders(self, image: np.ndarray, min_length: int = 30,
                        debug_prefix: str = 'debug') -> Dict:
        """
        STEP 2: Extract borders/lines using OpenCV
        Works with ANY color (red, blue, green, black, etc.)
        """
        print("-" * 30)
        print("STEP 2: BORDER EXTRACTION")
        print("-" * 30)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Edge detection (color-agnostic)
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)

        # Apply morphological dilation to strengthen faint or broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # SAVE EDGE DETECTION OUTPUT
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        edges_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_02a_edges_dilated.png")
        cv2.imwrite(edges_path, edges)
        print(f"✓ Saved edge detection (with dilation): {edges_path}")

        # Hough Line Transform - finds line segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=10
        )

        horizontal_lines = []
        vertical_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx ** 2 + dy ** 2)

                if length < min_length:
                    continue

                # Calculate angle
                angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)

                # Classify as horizontal or vertical
                if angle < 10 or angle > 170:  # Horizontal
                    y_avg = (y1 + y2) / 2
                    horizontal_lines.append({
                        'line_id': f"h_line_{len(horizontal_lines)}",
                        'type': 'horizontal',
                        'position': y_avg,
                        'start': min(x1, x2),
                        'end': max(x1, x2),
                        'endpoints': [(x1, y1), (x2, y2)],
                        'length': length
                    })

                elif 80 < angle < 100:  # Vertical
                    x_avg = (x1 + x2) / 2
                    vertical_lines.append({
                        'line_id': f"v_line_{len(vertical_lines)}",
                        'type': 'vertical',
                        'position': x_avg,
                        'start': min(y1, y2),
                        'end': max(y1, y2),
                        'endpoints': [(x1, y1), (x2, y2)],
                        'length': length
                    })

        # Merge nearby parallel lines
        horizontal_lines = self._merge_parallel_lines(horizontal_lines, tolerance=10)
        vertical_lines = self._merge_parallel_lines(vertical_lines, tolerance=10)

        # Sort by position
        horizontal_lines.sort(key=lambda l: l['position'])
        vertical_lines.sort(key=lambda l: l['position'])

        # SAVE BORDER VISUALIZATION
        border_vis = Image.fromarray(image)
        draw = ImageDraw.Draw(border_vis)

        for h_line in horizontal_lines:
            y = int(h_line['position'])
            x1 = int(h_line['start'])
            x2 = int(h_line['end'])
            draw.line([(x1, y), (x2, y)], fill=(255, 0, 0), width=3)

        for v_line in vertical_lines:
            x = int(v_line['position'])
            y1 = int(v_line['start'])
            y2 = int(v_line['end'])
            draw.line([(x, y1), (x, y2)], fill=(0, 0, 255), width=3)

        border_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_02b_borders.png")
        border_vis.save(border_path)
        print(f"✓ Saved border visualization: {border_path}")

        print(f"✓ Found {len(horizontal_lines)} horizontal borders")
        print(f"✓ Found {len(vertical_lines)} vertical borders")
        print()

        return {
            'horizontal': horizontal_lines,
            'vertical': vertical_lines
        }

    def _merge_parallel_lines(self, lines: List[Dict], tolerance: int = 10) -> List[Dict]:
        """Merge lines that are close and parallel"""
        if len(lines) == 0:
            return []

        lines.sort(key=lambda l: l['position'])

        merged = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            group = [line1]
            used[i] = True

            for j in range(i + 1, len(lines)):
                line2 = lines[j]
                if abs(line1['position'] - line2['position']) < tolerance:
                    group.append(line2)
                    used[j] = True
                elif line2['position'] - line1['position'] > tolerance:
                    break

            avg_position = np.mean([l['position'] for l in group])
            min_start = min([l['start'] for l in group])
            max_end = max([l['end'] for l in group])

            merged.append({
                'line_id': f"{line1['type']}_merged_{len(merged)}",
                'type': line1['type'],
                'position': avg_position,
                'start': min_start,
                'end': max_end,
                'length': max_end - min_start
            })

        return merged

    def extract_words(self, image: np.ndarray, debug_prefix: str = 'debug') -> List[Dict]:
        """
        STEP 3: Extract words with OCR + EXTENSIVE DEBUGGING
        """
        print("-" * 30)
        print("STEP 3: OCR TEXT EXTRACTION")
        print("-" * 30)

        self._load_ocr()

        # SAVE THE IMAGE BEING PROCESSED FOR DEBUGGING
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        debug_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_03a_ocr_input.png")
        cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved OCR input image: {debug_path}")
        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Image size: {image.shape[1]}x{image.shape[0]} pixels")

        # Check if image is too small
        if image.shape[0] < 20 or image.shape[1] < 20:
            print(f"⚠️  WARNING: Image too small ({image.shape[1]}x{image.shape[0]})! OCR may fail.")

        # Try OCR
        print("🔍 Running OCR...")
        result = self.ocr.ocr(image)

        # COMPLETE DEBUG - DUMP EVERYTHING
        print(f"\n{'=' * 60}")
        print("🔍 COMPLETE OCR DEBUG OUTPUT")
        print(f"{'=' * 60}")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if result else 'N/A'}")

        if result is None:
            print("✗ OCR returned None!")
            return []

        if len(result) == 0:
            print("✗ OCR returned empty list!")
            return []

        if result[0] is None:
            print("✗ OCR result[0] is None!")
            return []

        print(f"\nresult[0] type: {type(result[0])}")

        # DUMP THE ENTIRE result[0] STRUCTURE
        if isinstance(result[0], dict):
            print(f"\n📦 result[0] is a DICT with keys: {list(result[0].keys())}")
            for key, value in result[0].items():
                val_type = type(value)
                val_len = len(value) if hasattr(value, '__len__') and not isinstance(value,
                                                                                     (str, int, float)) else 'N/A'
                print(f"   '{key}': type={val_type}, len={val_len}")
                # Only show first item if it's a list/tuple
                if isinstance(value, (list, tuple)) and len(value) > 0 and len(value) < 100:
                    try:
                        print(f"      First item: {value[0]}")
                    except:
                        print(f"      First item: <cannot display>")
        elif isinstance(result[0], (list, tuple)):
            print(f"\n📦 result[0] is a LIST/TUPLE with {len(result[0])} items")
            if len(result[0]) > 0:
                print(f"   First item type: {type(result[0][0])}")
                print(f"   First item: {result[0][0]}")
                if len(result[0]) > 1:
                    print(f"   Second item type: {type(result[0][1])}")
                    print(f"   Second item: {result[0][1]}")
        else:
            print(f"\n❌ result[0] is UNEXPECTED TYPE: {type(result[0])}")
            print(f"   Value: {result[0]}")

        print(f"{'=' * 60}\n")

        # Handle different PaddleOCR return formats
        if isinstance(result[0], dict):
            print("🔍 Detected DICT format - searching for detections...")
            # Try common keys in order of priority
            possible_keys = [
                ('rec_texts', 'rec_polys', 'rec_scores'),  # New PaddleX format
                ('rec_texts', 'rec_boxes', 'rec_scores'),  # Alternative box format
                ('rec_res', None, None),  # Legacy format
                ('dt_polys', None, None),  # Detection only
            ]

            detections = None
            texts = None
            boxes = None
            scores = None

            # Check which format we have
            dict_keys = list(result[0].keys())
            print(f"   Available keys: {dict_keys}")

            # New PaddleX format with separate arrays
            if 'rec_texts' in result[0]:
                print("   ✓ Found 'rec_texts' key - using PaddleX format")
                texts = result[0]['rec_texts']
                boxes = result[0].get('rec_polys') or result[0].get('rec_boxes') or result[0].get('dt_polys')
                scores = result[0].get('rec_scores', [1.0] * len(texts))

                if boxes is None:
                    print("   ✗ No bounding boxes found!")
                    return []

                # Combine into standard format
                detections = []
                for i in range(len(texts)):
                    if i < len(boxes):
                        detections.append([boxes[i], (texts[i], scores[i] if i < len(scores) else 1.0)])

                print(f"   ✓ Combined {len(detections)} text+box pairs")

            # Legacy format
            elif 'rec_res' in result[0]:
                print("   ✓ Found 'rec_res' key")
                detections = result[0]['rec_res']

            if detections is None or len(detections) == 0:
                print(f"   ✗ Could not extract detections from keys: {dict_keys}")
                return []
        elif isinstance(result[0], (list, tuple)):
            print("🔍 Detected LIST format")
            detections = result[0]
        else:
            print(f"✗ Unexpected result[0] type: {type(result[0])}")
            return []

        print(f"✓ Found {len(detections)} detections to process\n")

        # Show sample detections with FULL structure
        if len(detections) > 0:
            print("📝 Sample detection structures:")
            for idx in range(min(2, len(detections))):
                print(f"\n  Detection {idx + 1}:")
                print(f"    Type: {type(detections[idx])}")
                print(f"    Value: {detections[idx]}")

        words = []
        word_id = 0

        print(f"\n🔄 Processing {len(detections)} detections...")

        for idx, line in enumerate(detections):
            try:
                # Handle dict format (newer PaddleOCR)
                if isinstance(line, dict):
                    if 'text' in line and 'bbox' in line:
                        text = line['text']
                        bbox_points = line['bbox']
                        confidence = line.get('score', 1.0)
                    else:
                        print(f"  ⚠️  Detection {idx}: dict missing 'text' or 'bbox' keys: {line.keys()}")
                        continue
                # Handle list/tuple format (older PaddleOCR)
                elif isinstance(line, (list, tuple)) and len(line) >= 2:
                    bbox_points = line[0]
                    text_conf = line[1]

                    if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                        text, confidence = text_conf
                    elif isinstance(text_conf, str):
                        text, confidence = text_conf, 1.0
                    else:
                        print(f"  ⚠️  Detection {idx}: unexpected text_conf format: {type(text_conf)}")
                        continue
                else:
                    print(
                        f"  ⚠️  Detection {idx}: unexpected format - type={type(line)}, len={len(line) if hasattr(line, '__len__') else 'N/A'}")
                    continue

                # Skip empty text
                if not text or (isinstance(text, str) and text.strip() == ''):
                    print(f"  ⚠️  Detection {idx}: empty text, skipping")
                    continue

                # Extract bounding box coordinates
                # Handle numpy arrays or lists
                if isinstance(bbox_points, np.ndarray):
                    bbox_points = bbox_points.tolist()

                if isinstance(bbox_points, (list, tuple)) and len(bbox_points) >= 4:
                    # bbox_points should be [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    try:
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                    except (IndexError, TypeError) as e:
                        print(f"  ⚠️  Detection {idx}: bbox coordinate extraction failed: {e}")
                        continue
                else:
                    print(f"  ⚠️  Detection {idx}: invalid bbox format: {bbox_points}")
                    continue

                words.append({
                    'word_id': f"word_{word_id}",
                    'text': text,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'confidence': float(confidence)
                })
                word_id += 1

            except Exception as e:
                print(f"  ❌ Detection {idx} failed: {e}")
                continue

        print(f"\n✓ Successfully extracted {len(words)} words from {len(detections)} detections")

        # SAVE WORD VISUALIZATION
        word_vis = Image.fromarray(image)
        draw = ImageDraw.Draw(word_vis)

        for word in words:
            x1, y1, x2, y2 = word['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            cx, cy = word['center']
            r = 3
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0))

        word_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_03b_words.png")
        word_vis.save(word_path)
        print(f"✓ Saved word visualization: {word_path}")
        print(f"✓ Extracted {len(words)} words")

        if len(words) > 0:
            print(f"  Sample words: {[w['text'] for w in words[:5]]}")
        else:
            print("⚠️  NO WORDS DETECTED! Check:")
            print("     1. Is the table region too small?")
            print("     2. Is the text readable in the saved image?")
            print("     3. Try adjusting OCR parameters")

        print()
        return words

    def map_words_to_borders(self, words: List[Dict], borders: Dict) -> List[Dict]:
        """
        STEP 4: Map each word to surrounding borders
        """
        print("-" * 30)
        print("STEP 4: WORD-TO-BORDER MAPPING")
        print("-" * 30)

        h_lines = borders['horizontal']
        v_lines = borders['vertical']

        mappings = []

        for word in words:
            x1, y1, x2, y2 = word['bbox']
            cx, cy = word['center']

            top_line = self._find_closest_line(cy, h_lines, 'above')
            bottom_line = self._find_closest_line(cy, h_lines, 'below')
            left_line = self._find_closest_line(cx, v_lines, 'left')
            right_line = self._find_closest_line(cx, v_lines, 'right')

            mapping = {
                'word_id': word['word_id'],
                'word_text': word['text'],
                'word_bbox': word['bbox'],
                'word_center': word['center'],
                'borders': {}
            }

            if top_line:
                mapping['borders']['top'] = {
                    'line_id': top_line['line_id'],
                    'position': top_line['position'],
                    'distance': abs(cy - top_line['position'])
                }

            if bottom_line:
                mapping['borders']['bottom'] = {
                    'line_id': bottom_line['line_id'],
                    'position': bottom_line['position'],
                    'distance': abs(cy - bottom_line['position'])
                }

            if left_line:
                mapping['borders']['left'] = {
                    'line_id': left_line['line_id'],
                    'position': left_line['position'],
                    'distance': abs(cx - left_line['position'])
                }

            if right_line:
                mapping['borders']['right'] = {
                    'line_id': right_line['line_id'],
                    'position': right_line['position'],
                    'distance': abs(cx - right_line['position'])
                }

            if top_line and bottom_line and left_line and right_line:
                mapping['cell_defined_by'] = tuple(sorted([
                    top_line['line_id'],
                    bottom_line['line_id'],
                    left_line['line_id'],
                    right_line['line_id']
                ]))

            mappings.append(mapping)

        print(f"✓ Created {len(mappings)} word-to-border mappings")
        print()
        return mappings

    def _find_closest_line(self, position: float, lines: List[Dict],
                           direction: str) -> Optional[Dict]:
        """Find closest line in a direction"""
        candidates = []

        for line in lines:
            line_pos = line['position']

            if direction == 'above' and line_pos < position:
                candidates.append((position - line_pos, line))
            elif direction == 'below' and line_pos > position:
                candidates.append((line_pos - position, line))
            elif direction == 'left' and line_pos < position:
                candidates.append((position - line_pos, line))
            elif direction == 'right' and line_pos > position:
                candidates.append((line_pos - position, line))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return None

    def build_border_graph(self, borders: Dict) -> nx.Graph:
        """STEP 5: Build Border Graph"""
        print("-" * 30)
        print("STEP 5: BUILD BORDER GRAPH")
        print("-" * 30)

        G = nx.Graph()

        h_lines = borders['horizontal']
        v_lines = borders['vertical']

        for line in h_lines + v_lines:
            G.add_node(
                line['line_id'],
                type=line['type'],
                position=line['position'],
                start=line['start'],
                end=line['end'],
                length=line['length'],
                node_type='border'
            )

        for h_line in h_lines:
            for v_line in v_lines:
                h_y = h_line['position']
                h_x_start, h_x_end = h_line['start'], h_line['end']

                v_x = v_line['position']
                v_y_start, v_y_end = v_line['start'], v_line['end']

                tolerance = 2
                if (h_x_start - tolerance <= v_x <= h_x_end + tolerance and
                        v_y_start - tolerance <= h_y <= v_y_end + tolerance):
                    G.add_edge(
                        h_line['line_id'],
                        v_line['line_id'],
                        intersection_point=(v_x, h_y),
                        edge_type='intersection'
                    )

        print(f"✓ Border Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print()
        return G

    def build_word_graph(self, words: List[Dict], k: int = 4) -> nx.DiGraph:
        """STEP 6: Build Word Graph"""
        print("-" * 30)
        print("STEP 6: BUILD WORD GRAPH")
        print("-" * 30)

        G = nx.DiGraph()

        for word in words:
            G.add_node(
                word['word_id'],
                text=word['text'],
                bbox=word['bbox'],
                center=word['center'],
                confidence=word['confidence'],
                node_type='word'
            )

        if len(words) == 0:
            print("✗ No words to build graph from")
            return G

        if len(words) == 1:
            print("✓ Word Graph: 1 node, 0 edges (single word)")
            return G

        word_centers = np.array([w['center'] for w in words])
        word_ids = [w['word_id'] for w in words]

        for i, w1 in enumerate(words):
            if not word_ids or i >= len(word_centers):
                continue

            distances = np.sqrt(np.sum((word_centers - word_centers[i]) ** 2, axis=1))

            # Adjust k if we have fewer words than k+1
            actual_k = min(k, len(words) - 1)
            if actual_k == 0:
                continue

            k_indices = np.argpartition(distances, actual_k)[1:actual_k + 1]

            for j_idx in k_indices:
                if j_idx >= len(words):
                    continue

                w2 = words[j_idx]
                dist = distances[j_idx]

                dx = w2['center'][0] - w1['center'][0]
                dy = w2['center'][1] - w1['center'][1]

                if abs(dx) > abs(dy):
                    direction = 'horizontal'
                    edge_type = 'right' if dx > 0 else 'left'
                else:
                    direction = 'vertical'
                    edge_type = 'down' if dy > 0 else 'up'

                G.add_edge(w1['word_id'], w2['word_id'],
                           direction=direction, edge_type=edge_type, distance=dist)

        print(f"✓ Word Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print()
        return G

    def infer_cells(self, mappings: List[Dict], debug_prefix: str = 'debug') -> List[Dict]:
        """STEP 7: Infer cells"""
        print("-" * 30)
        print("STEP 7: CELL INFERENCE")
        print("-" * 30)

        cell_groups = defaultdict(list)

        for mapping in mappings:
            if 'cell_defined_by' in mapping:
                signature = mapping['cell_defined_by']
                cell_groups[signature].append(mapping)

        cells = []
        for cell_id, (signature, word_mappings) in enumerate(cell_groups.items()):
            x1_coords = [m['word_bbox'][0] for m in word_mappings]
            y1_coords = [m['word_bbox'][1] for m in word_mappings]
            x2_coords = [m['word_bbox'][2] for m in word_mappings]
            y2_coords = [m['word_bbox'][3] for m in word_mappings]

            left_pos = min(x1_coords)
            top_pos = min(y1_coords)
            right_pos = max(x2_coords)
            bottom_pos = max(y2_coords)

            word_mappings.sort(key=lambda m: (m['word_center'][1], m['word_center'][0]))

            cells.append({
                'cell_id': f"cell_{cell_id}",
                'border_signature': signature,
                'bbox_word_aggregate': [int(left_pos), int(top_pos), int(right_pos), int(bottom_pos)],
                'center': [(left_pos + right_pos) / 2, (top_pos + bottom_pos) / 2],
                'word_ids': [m['word_id'] for m in word_mappings],
                'text': ' '.join([m['word_text'] for m in word_mappings])
            })

        print(f"✓ Inferred {len(cells)} cells")
        print()
        return cells

    def visualize_all_combined(self, image: np.ndarray, borders: Dict,
                               cells: List[Dict], words: List[Dict],
                               output_path: str = 'viz_all.png'):
        """Visualize everything together"""
        print("-" * 30)
        print("FINAL VISUALIZATION")
        print("-" * 30)

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        np.random.seed(42)
        for cell in cells:
            x1, y1, x2, y2 = cell['bbox_word_aggregate']
            color = (np.random.randint(150, 255),
                     np.random.randint(150, 255),
                     np.random.randint(150, 255))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        for h_line in borders['horizontal']:
            y = int(h_line['position'])
            x1 = int(h_line['start'])
            x2 = int(h_line['end'])
            draw.line([(x1, y), (x2, y)], fill=(255, 0, 0), width=2)

        for v_line in borders['vertical']:
            x = int(v_line['position'])
            y1 = int(v_line['start'])
            y2 = int(v_line['end'])
            draw.line([(x, y1), (x, y2)], fill=(0, 0, 255), width=2)

        for word in words:
            x1, y1, x2, y2 = word['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=1)

            cx, cy = word['center']
            r = 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0))

        img.save(output_path)
        print(f"✓ Saved final visualization: {output_path}")
        print()
        return img

    def _process_single_table_region(self, table_img: np.ndarray,
                                     output_prefix: str, table_index: int) -> Optional[Dict]:
        """Process a single table region"""
        print(f"\n--- Processing Table Region {table_index} ---")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        table_save_path = os.path.join(BASE_DIR, "out", f"{output_prefix}_table{table_index}_00_region.png")
        os.makedirs(os.path.dirname(table_save_path), exist_ok=True)
        cv2.imwrite(table_save_path, cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved table region: {table_save_path}")

        current_prefix = f"{output_prefix}_table{table_index}"

        borders = self.extract_borders(table_img, debug_prefix=current_prefix)
        words = self.extract_words(table_img, debug_prefix=current_prefix)

        if len(words) == 0:
            print(f"⚠️  WARNING: No words detected for Table {table_index}!")
            print(f"   Check the saved OCR input: {current_prefix}_03a_ocr_input.png")
            # Continue anyway to show borders

        mappings = self.map_words_to_borders(words, borders)
        G_borders = self.build_border_graph(borders)
        G_words = self.build_word_graph(words)
        cells = self.infer_cells(mappings, debug_prefix=current_prefix)

        result = {
            'table_index': table_index,
            'table_image_shape': table_img.shape[:2],
            'borders': borders,
            'words': words,
            'cells': cells,
            'border_graph': G_borders,
            'word_graph': G_words,
            'word_border_mappings': mappings,
            'stats': {
                'num_h_lines': len(borders['horizontal']),
                'num_v_lines': len(borders['vertical']),
                'num_words': len(words),
                'num_cells': len(cells)
            }
        }

        final_path = os.path.join(BASE_DIR, "out", f"{current_prefix}_99_final.png")
        self.visualize_all_combined(table_img, borders, cells, words, final_path)

        print(f"--- Table {table_index} Processing Complete ---")
        return result

    def process_table(self, image_path: str, output_prefix: str = 'output',
                      skip_table_detection: bool = False, visualize: bool = True,
                      detection_threshold: float = 0.03) -> List[Dict[str, Any]]:
        """Complete pipeline - NOW RETURNS TABLE COORDS IMMEDIATELY AFTER DETECTION"""
        print("=" * 70)
        print(f"PROCESSING: {image_path}")
        print("=" * 70 + "\n")

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        all_results = []

        if not skip_table_detection:
            # STEP 1: GET TABLE COORDINATES IMMEDIATELY
            tables = self.detect_tables(image_pil, debug_prefix=output_prefix,
                                        detection_threshold=detection_threshold)

            if len(tables) == 0:
                print("✗ No tables detected! Processing full image as one table.")
                result = self._process_single_table_region(image_np, output_prefix, table_index=1)
                if result:
                    all_results.append(result)
            else:
                # Process each detected table
                for i, table in enumerate(tables):
                    print(f"\n{'=' * 70}")
                    print(f"PROCESSING TABLE {i + 1}/{len(tables)}")
                    print(f"BBox: {table['bbox']}, Size: {table['width']}x{table['height']}")
                    print(f"{'=' * 70}")

                    x1, y1, x2, y2 = table['bbox']
                    table_img = image_np[y1:y2, x1:x2]

                    result = self._process_single_table_region(table_img, output_prefix, table_index=i + 1)
                    if result:
                        # Add original table detection info to result
                        result['detected_bbox'] = table['bbox']
                        result['detection_confidence'] = table['confidence']
                        all_results.append(result)
        else:
            print("⊗ Skipping table detection - processing full image.")
            result = self._process_single_table_region(image_np, output_prefix, table_index=1)
            if result:
                all_results.append(result)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE - FINAL SUMMARY")
        print("=" * 70)
        print(f"Total tables processed: {len(all_results)}\n")

        for result in all_results:
            print(f"  📊 Table {result['table_index']}:")
            if 'detected_bbox' in result:
                print(f"     - Detection BBox: {result['detected_bbox']}")
                print(f"     - Confidence: {result['detection_confidence']:.3f}")
            print(f"     - Words extracted: {result['stats']['num_words']}")
            print(f"     - Cells inferred: {result['stats']['num_cells']}")
            if result['stats']['num_words'] == 0:
                print(f"     ⚠️  NO WORDS - Check OCR input image!")
        print()

        return all_results

    def export_json(self, result_list: List[Dict[str, Any]], output_prefix: str):
        """Export results to JSON"""

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(BASE_DIR, "out")
        os.makedirs(output_dir, exist_ok=True)

        for result in result_list:
            table_idx = result['table_index']
            output_path = os.path.join(output_dir, f"{output_prefix}_table{table_idx}_data.json")

            data = {
                'table_index': table_idx,
                'table_image_shape': result['table_image_shape'],
                'detected_bbox': result.get('detected_bbox'),
                'detection_confidence': result.get('detection_confidence'),
                'stats': convert(result['stats']),
                'borders': convert(result['borders']),
                'word_to_border_mappings': convert(result['word_border_mappings']),
                'inferred_cells': convert(result['cells']),
                'border_graph': {
                    'nodes': [convert({'id': n, **a}) for n, a in result['border_graph'].nodes(data=True)],
                    'edges': [convert({'source': u, 'target': v, **a})
                              for u, v, a in result['border_graph'].edges(data=True)]
                },
                'word_graph': {
                    'nodes': [convert({'id': n, **a}) for n, a in result['word_graph'].nodes(data=True)],
                    'edges': [convert({'source': u, 'target': v, **a})
                              for u, v, a in result['word_graph'].edges(data=True)]
                },
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"✓ Exported JSON for Table {table_idx} to: {output_path}")
        print()