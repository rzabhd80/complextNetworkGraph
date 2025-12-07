"""
Hybrid Approach: Table Transformer + Border Extraction + Visualizations
========================================================================

Pipeline:
1. Table Transformer DETECTION → Find table regions (works with colored tables!)
2. OpenCV Border Extraction → Extract all lines/borders (ANY color)
3. OCR → Extract words
4. Map words to borders → Which words between which lines
5. Build 3 graphs: Border Graph, Word Graph, Word-Border Mapping
6. VISUALIZE: Borders, Cells, Words, and Combined views

ENHANCED WITH STEP-BY-STEP DEBUG OUTPUTS
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
from typing import List, Dict, Tuple, Optional
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
        """Load OCR"""
        if self.ocr is None:
            print("Loading OCR...")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',

            )
            print("✓ OCR loaded\n")

    def detect_tables(self, image: Image.Image, debug_prefix: str = 'debug') -> List[Dict]:
        """
        STEP 1: Use Table Transformer to detect table REGIONS
        This works even with colored borders!
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

        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.7
        )[0]

        # Extract table regions
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == 0:  # Table class
                tables.append({
                    'bbox': [int(i) for i in box.tolist()],
                    'confidence': score.item(),
                    'detection_method': 'table_transformer'
                })

        # SAVE DETECTION VISUALIZATION
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        debug_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_01_table_detection.png")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)
        for i, table in enumerate(tables):
            x1, y1, x2, y2 = table['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
            draw.text((x1, y1 - 20), f"Table {i + 1} (conf:{table['confidence']:.2f})", fill='red')
        debug_img.save(debug_path)
        print(f"✓ Saved detection visualization: {debug_path}")
        print(f"✓ Found {len(tables)} table(s)")

        if len(tables) > 0:
            for i, table in enumerate(tables):
                print(f"  Table {i + 1}: bbox={table['bbox']}, confidence={table['confidence']:.3f}")

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
        print("=" * 70)
        print("STEP 2: BORDER EXTRACTION")
        print("=" * 70)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Edge detection (color-agnostic)
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)

        # SAVE EDGE DETECTION OUTPUT
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        edges_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_02a_edges.png")
        cv2.imwrite(edges_path, edges)
        print(f"✓ Saved edge detection: {edges_path}")

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

        merged = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            group = [line1]
            used[i] = True

            for j, line2 in enumerate(lines):
                if i != j and not used[j]:
                    if abs(line1['position'] - line2['position']) < tolerance:
                        group.append(line2)
                        used[j] = True

            # Average the group
            avg_position = np.mean([l['position'] for l in group])
            min_start = min([l['start'] for l in group])
            max_end = max([l['end'] for l in group])

            merged.append({
                'line_id': line1['line_id'],
                'type': line1['type'],
                'position': avg_position,
                'start': min_start,
                'end': max_end,
                'length': max_end - min_start
            })

        return merged

    def extract_words(self, image: np.ndarray, debug_prefix: str = 'debug') -> List[Dict]:
        """
        STEP 3: Extract words with OCR
        """
        print("=" * 70)
        print("STEP 3: OCR TEXT EXTRACTION")
        print("=" * 70)

        self._load_ocr()

        # SAVE THE IMAGE BEING PROCESSED FOR DEBUGGING
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        debug_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_03a_ocr_input.png")
        cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved OCR input image: {debug_path}")
        print(f"✓ Image shape: {image.shape}")

        result = self.ocr.ocr(image)

        # DEBUG: Print raw OCR result structure
        print(f"✓ Raw OCR result type: {type(result)}")

        if result is None:
            print("✗ OCR returned None!")
            return []

        if len(result) == 0:
            print("✗ OCR returned empty list!")
            return []

        if result[0] is None:
            print("✗ OCR result[0] is None!")
            return []

        print(f"✓ OCR detected {len(result[0])} text regions")

        words = []
        word_id = 0

        if result and len(result) > 0:
            for line in result[0]:  # result[0] = first page
                # Safe unpacking: bbox is first element, text/conf is second element
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue  # skip malformed entries

                bbox = line[0]
                # Handle different formats for text/conf
                text_conf = line[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                    text, confidence = text_conf
                elif isinstance(text_conf, str):
                    text, confidence = text_conf, 1.0  # fallback confidence
                else:
                    continue  # skip unknown format

                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)

                words.append({
                    'word_id': f"word_{word_id}",
                    'text': text,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'confidence': confidence
                })
                word_id += 1

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

        print()
        return words

    def map_words_to_borders(self, words: List[Dict], borders: Dict) -> List[Dict]:
        """
        STEP 4: Map each word to surrounding borders
        Returns which borders (lines) surround each word
        """
        print("=" * 70)
        print("STEP 4: WORD-TO-BORDER MAPPING")
        print("=" * 70)

        h_lines = borders['horizontal']
        v_lines = borders['vertical']

        mappings = []

        for word in words:
            x1, y1, x2, y2 = word['bbox']
            cx, cy = word['center']

            # Find closest borders in each direction
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

            # Cell defined by 4 borders
            if top_line and bottom_line and left_line and right_line:
                mapping['cell_defined_by'] = [
                    top_line['line_id'],
                    bottom_line['line_id'],
                    left_line['line_id'],
                    right_line['line_id']
                ]

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
        """
        STEP 5: Build Border Graph
        Nodes = lines, Edges = intersections
        """
        print("=" * 70)
        print("STEP 5: BUILD BORDER GRAPH")
        print("=" * 70)

        G = nx.Graph()

        h_lines = borders['horizontal']
        v_lines = borders['vertical']

        # Add nodes
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

        # Add edges (intersections)
        for h_line in h_lines:
            for v_line in v_lines:
                h_y = h_line['position']
                h_x_start, h_x_end = h_line['start'], h_line['end']

                v_x = v_line['position']
                v_y_start, v_y_end = v_line['start'], v_line['end']

                # Check intersection
                if (h_x_start <= v_x <= h_x_end and
                        v_y_start <= h_y <= v_y_end):
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
        """
        STEP 6: Build Word Graph
        Nodes = words, Edges = spatial proximity
        """
        print("=" * 70)
        print("STEP 6: BUILD WORD GRAPH")
        print("=" * 70)

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

        for i, w1 in enumerate(words):
            distances = []
            for j, w2 in enumerate(words):
                if i != j:
                    dist = np.sqrt((w2['center'][0] - w1['center'][0]) ** 2 +
                                   (w2['center'][1] - w1['center'][1]) ** 2)
                    distances.append((j, dist, w2))

            distances.sort(key=lambda x: x[1])

            for j, dist, w2 in distances[:k]:
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
        """
        STEP 7: Infer cells from word-border mappings
        """
        print("=" * 70)
        print("STEP 7: CELL INFERENCE")
        print("=" * 70)

        cell_groups = defaultdict(list)

        for mapping in mappings:
            if 'cell_defined_by' in mapping:
                signature = tuple(sorted(mapping['cell_defined_by']))
                cell_groups[signature].append(mapping)

        cells = []
        for cell_id, (signature, word_mappings) in enumerate(cell_groups.items()):
            top_pos = min([m['borders']['top']['position'] for m in word_mappings
                           if 'top' in m['borders']], default=0)
            bottom_pos = max([m['borders']['bottom']['position'] for m in word_mappings
                              if 'bottom' in m['borders']], default=0)
            left_pos = min([m['borders']['left']['position'] for m in word_mappings
                            if 'left' in m['borders']], default=0)
            right_pos = max([m['borders']['right']['position'] for m in word_mappings
                             if 'right' in m['borders']], default=0)

            cells.append({
                'cell_id': f"cell_{cell_id}",
                'border_signature': signature,
                'bbox': [int(left_pos), int(top_pos), int(right_pos), int(bottom_pos)],
                'center': [(left_pos + right_pos) / 2, (top_pos + bottom_pos) / 2],
                'word_ids': [m['word_id'] for m in word_mappings],
                'text': ' '.join([m['word_text'] for m in word_mappings])
            })

        print(f"✓ Inferred {len(cells)} cells")
        print()
        return cells

    # ============================================
    # VISUALIZATION METHODS
    # ============================================

    def visualize_all_combined(self, image: np.ndarray, borders: Dict,
                               cells: List[Dict], words: List[Dict],
                               output_path: str = 'viz_all.png'):
        """Visualize everything together"""
        print("=" * 70)
        print("FINAL VISUALIZATION")
        print("=" * 70)

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        # 1. Draw cells (filled, semi-transparent effect with lighter colors)
        np.random.seed(42)
        for cell in cells:
            x1, y1, x2, y2 = cell['bbox']
            color = (np.random.randint(150, 255),
                     np.random.randint(150, 255),
                     np.random.randint(150, 255))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 2. Draw borders (horizontal=RED, vertical=BLUE)
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

        # 3. Draw words (GREEN boxes with RED centers)
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

    # ============================================
    # MAIN PIPELINE
    # ============================================

    def process_table(self, image_path: str, output_prefix: str = 'output',
                      skip_table_detection: bool = False) -> Optional[Dict]:
        """Complete pipeline with step-by-step debug output"""
        print("=" * 70)
        print(f"PROCESSING: {image_path}")
        print("=" * 70 + "\n")

        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        # Step 1: Table Transformer detects table REGION (or skip)
        if not skip_table_detection:
            tables = self.detect_tables(image_pil, debug_prefix=output_prefix)

            if len(tables) == 0:
                print("✗ No tables detected! Trying full image instead...")
                table_img = image_np
            else:
                # Crop to table
                table = tables[0]
                x1, y1, x2, y2 = table['bbox']
                table_img = image_np[y1:y2, x1:x2]
        else:
            print("⊗ Skipping table detection - using full image")
            table_img = image_np

        # Save the cropped/full table image
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        table_save_path = os.path.join(BASE_DIR, "out", f"{output_prefix}_00_table_region.png")
        os.makedirs(os.path.dirname(table_save_path), exist_ok=True)
        cv2.imwrite(table_save_path, cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved table region: {table_save_path}\n")

        # Step 2: Extract borders (OpenCV)
        borders = self.extract_borders(table_img, debug_prefix=output_prefix)

        # Step 3: Extract words (OCR)
        words = self.extract_words(table_img, debug_prefix=output_prefix)

        if len(words) == 0:
            print("✗ No words detected! Cannot continue.")
            return None

        # Step 4: Map words to borders
        mappings = self.map_words_to_borders(words, borders)

        # Step 5: Build border graph
        G_borders = self.build_border_graph(borders)

        # Step 6: Build word graph
        G_words = self.build_word_graph(words)

        # Step 7: Infer cells
        cells = self.infer_cells(mappings, debug_prefix=output_prefix)

        result = {
            'table_image': table_img,
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

        # Final combined visualization
        final_path = os.path.join(BASE_DIR, "out", f"{output_prefix}_99_final.png")
        self.visualize_all_combined(table_img, borders, cells, words, final_path)

        print("=" * 70)
        print("PIPELINE COMPLETE - SUMMARY")
        print("=" * 70)
        print(f"✓ Horizontal borders: {len(borders['horizontal'])}")
        print(f"✓ Vertical borders: {len(borders['vertical'])}")
        print(f"✓ Words extracted: {len(words)}")
        print(f"✓ Cells inferred: {len(cells)}")
        print(f"✓ Border graph: {G_borders.number_of_nodes()} nodes, {G_borders.number_of_edges()} edges")
        print(f"✓ Word graph: {G_words.number_of_nodes()} nodes, {G_words.number_of_edges()} edges")
        print()

        return result

    def export_json(self, result: Dict, output_path: str):
        """Export everything to JSON"""

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

        data = {
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
            'word_to_border_mappings': convert(result['word_border_mappings']),
            'inferred_cells': convert(result['cells']),
            'borders': convert(result['borders']),
            'stats': convert(result['stats'])
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Exported JSON to: {output_path}")