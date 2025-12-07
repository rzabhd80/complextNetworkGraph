"""
YOLO-Based Table Extractor with Robust Table Detection
=====================================================

- Detects tables inside pages, avoiding full-page detections
- Uses YOLO for candidate regions
- Uses OpenCV for line/border detection
- OCR via PaddleOCR
- Word-to-border mapping, cell inference, and visualization
"""

import os
import gc
import json
import warnings
from collections import defaultdict
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
import networkx as nx
from paddleocr import PaddleOCR
from ultralytics import YOLO

warnings.filterwarnings('ignore')


class YOLOTableExtractor:
    def __init__(self, yolo_model: str = 'yolov8x.pt'):
        self.yolo_model_name = yolo_model
        self.yolo_model = None
        self.ocr = None
        print(f"YOLO Model: {yolo_model}\n")

    def _load_yolo(self):
        if self.yolo_model is None:
            print(f"Loading YOLO model: {self.yolo_model_name}...")
            self.yolo_model = YOLO(self.yolo_model_name)
            print("✓ YOLO model loaded\n")

    def _unload_yolo(self):
        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None
            gc.collect()
            print("✓ YOLO model unloaded\n")

    def _load_ocr(self):
        if self.ocr is None:
            print("Loading OCR...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            print("✓ OCR loaded\n")

    def detect_tables(self, image: Image.Image, debug_prefix: str = 'debug',
                      confidence_threshold: float = 0.25,
                      max_area_ratio: float = 0.9,
                      min_area_ratio: float = 0.01,
                      min_lines: int = 2) -> List[Dict]:
        """
        Detect tables inside a page image, avoiding full-page detection
        """
        print("=" * 70)
        print("STEP 1: TABLE DETECTION (YOLO + Line Verification)")
        print("=" * 70)

        self._load_yolo()
        image_np = np.array(image)
        results = self.yolo_model(image_np, conf=confidence_threshold, verbose=False)
        tables = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)
                image_area = image.width * image.height

                # Only accept plausible table regions
                if class_name not in ['dining table', 'book', 'laptop']:
                    continue
                if area < min_area_ratio * image_area or area > max_area_ratio * image_area:
                    continue

                # Optional: verify table lines inside candidate
                candidate_img = image_np[y1:y2, x1:x2]
                borders = self.extract_borders(candidate_img, visualize=False)
                if len(borders['horizontal']) < min_lines or len(borders['vertical']) < min_lines:
                    continue  # skip regions with too few lines

                tables.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class': class_name,
                    'detection_method': 'yolo_verified',
                    'area': area
                })

        # Sort tables by area (largest first)
        tables.sort(key=lambda t: t['area'], reverse=True)

        # Save debug visualization
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        debug_path = os.path.join(BASE_DIR, "out", f"{debug_prefix}_01_yolo_detection.png")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, table in enumerate(tables):
            x1, y1, x2, y2 = table['bbox']
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            label = f"{table['class']} {i+1}\nconf:{table['confidence']:.2f}\narea:{table['area']}"
            draw.text((x1, y1 - 60), label, fill=color)

        debug_img.save(debug_path)
        print(f"✓ Saved YOLO detection visualization: {debug_path}")
        print(f"✓ Found {len(tables)} potential table(s)\n")
        return tables

    def extract_borders(self, image: np.ndarray, min_length: int = 30, debug_prefix: str = 'debug',
                        visualize: bool = True) -> Dict:
        """Detect horizontal and vertical lines in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=50, minLineLength=min_length, maxLineGap=10)

        horizontal_lines, vertical_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx ** 2 + dy ** 2)
                if length < min_length:
                    continue
                angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
                if angle < 10 or angle > 170:  # horizontal
                    y_avg = (y1 + y2) / 2
                    horizontal_lines.append({'line_id': f"h_{len(horizontal_lines)}", 'type': 'horizontal',
                                             'position': y_avg, 'start': min(x1, x2), 'end': max(x1, x2),
                                             'length': length})
                elif 80 < angle < 100:  # vertical
                    x_avg = (x1 + x2) / 2
                    vertical_lines.append({'line_id': f"v_{len(vertical_lines)}", 'type': 'vertical',
                                           'position': x_avg, 'start': min(y1, y2), 'end': max(y1, y2),
                                           'length': length})

        horizontal_lines = self._merge_parallel_lines(horizontal_lines)
        vertical_lines = self._merge_parallel_lines(vertical_lines)

        horizontal_lines.sort(key=lambda l: l['position'])
        vertical_lines.sort(key=lambda l: l['position'])

        if visualize:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            border_vis = Image.fromarray(image)
            draw = ImageDraw.Draw(border_vis)
            for h in horizontal_lines:
                y = int(h['position'])
                draw.line([(int(h['start']), y), (int(h['end']), y)], fill=(255, 0, 0), width=3)
            for v in vertical_lines:
                x = int(v['position'])
                draw.line([(x, int(v['start'])), (x, int(v['end']))], fill=(0, 0, 255), width=3)
            border_path = os.path.join(BASE_DIR, f"{debug_prefix}_02_borders.png")
            border_vis.save(border_path)
            print(f"✓ Saved border visualization: {border_path}")

        return {'horizontal': horizontal_lines, 'vertical': vertical_lines}

    def _merge_parallel_lines(self, lines: List[Dict], tolerance: int = 10) -> List[Dict]:
        merged, used = [], [False] * len(lines)
        for i, line1 in enumerate(lines):
            if used[i]: continue
            group = [line1]; used[i] = True
            for j, line2 in enumerate(lines):
                if i != j and not used[j] and abs(line1['position'] - line2['position']) < tolerance:
                    group.append(line2); used[j] = True
            avg_pos = np.mean([l['position'] for l in group])
            min_start = min([l['start'] for l in group])
            max_end = max([l['end'] for l in group])
            merged.append({**line1, 'position': avg_pos, 'start': min_start, 'end': max_end, 'length': max_end - min_start})
        return merged

    # --- Full remaining pipeline: OCR, word mapping, graph, cell inference, visualization ---

    def extract_words(self, image: np.ndarray, debug_prefix: str = 'debug') -> List[Dict]:
        print("="*70); print("STEP 3: OCR TEXT EXTRACTION"); print("="*70)
        self._load_ocr()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        debug_path = os.path.join(BASE_DIR, f"{debug_prefix}_03a_ocr_input.png")
        cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = self.ocr.ocr(image)
        words = []; word_id = 0
        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                if not isinstance(line, (list, tuple)) or len(line) < 2: continue
                bbox = line[0]; text_conf = line[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                    text, confidence = text_conf
                elif isinstance(text_conf, str): text, confidence = text_conf, 1.0
                else: continue
                x_coords = [p[0] for p in bbox]; y_coords = [p[1] for p in bbox]
                x1, y1 = min(x_coords), min(y_coords); x2, y2 = max(x_coords), max(y_coords)
                words.append({'word_id': f"word_{word_id}", 'text': text, 'bbox': [int(x1), int(y1), int(x2), int(y2)],
                              'center': [(x1 + x2)/2, (y1 + y2)/2], 'confidence': confidence})
                word_id += 1

        word_vis = Image.fromarray(image); draw = ImageDraw.Draw(word_vis)
        for word in words:
            x1, y1, x2, y2 = word['bbox']; draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
            cx, cy = word['center']; r=3; draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255,0,0))
        word_path = os.path.join(BASE_DIR, f"{debug_prefix}_03b_words.png"); word_vis.save(word_path)
        print(f"✓ Saved word visualization: {word_path}")
        return words

    def map_words_to_borders(self, words: List[Dict], borders: Dict) -> List[Dict]:
        print("="*70); print("STEP 4: WORD-TO-BORDER MAPPING"); print("="*70)
        h_lines, v_lines = borders['horizontal'], borders['vertical']; mappings=[]
        for word in words:
            x1, y1, x2, y2 = word['bbox']; cx, cy = word['center']
            top_line = self._find_closest_line(cy, h_lines, 'above')
            bottom_line = self._find_closest_line(cy, h_lines, 'below')
            left_line = self._find_closest_line(cx, v_lines, 'left')
            right_line = self._find_closest_line(cx, v_lines, 'right')
            mapping = {'word_id': word['word_id'], 'word_text': word['text'], 'word_bbox': word['bbox'],
                       'word_center': word['center'], 'borders': {}}
            if top_line: mapping['borders']['top'] = {'line_id': top_line['line_id'],
                                                      'position': top_line['position'],
                                                      'distance': abs(cy - top_line['position'])}
            if bottom_line: mapping['borders']['bottom'] = {'line_id': bottom_line['line_id'],
                                                            'position': bottom_line['position'],
                                                            'distance': abs(cy - bottom_line['position'])}
            if left_line: mapping['borders']['left'] = {'line_id': left_line['line_id'],
                                                        'position': left_line['position'],
                                                        'distance': abs(cx - left_line['position'])}
            if right_line: mapping['borders']['right'] = {'line_id': right_line['line_id'],
                                                          'position': right_line['position'],
                                                          'distance': abs(cx - right_line['position'])}
            if top_line and bottom_line and left_line and right_line:
                mapping['cell_defined_by'] = [top_line['line_id'], bottom_line['line_id'],
                                              left_line['line_id'], right_line['line_id']]
            mappings.append(mapping)
        print(f"✓ Created {len(mappings)} word-to-border mappings\n")
        return mappings

    def _find_closest_line(self, position: float, lines: List[Dict], direction: str) -> Optional[Dict]:
        candidates = []
        for line in lines:
            line_pos = line['position']
            if direction == 'above' and line_pos < position: candidates.append((position-line_pos, line))
            elif direction == 'below' and line_pos > position: candidates.append((line_pos-position, line))
            elif direction == 'left' and line_pos < position: candidates.append((position-line_pos, line))
            elif direction == 'right' and line_pos > position: candidates.append((line_pos-position, line))
        if candidates: candidates.sort(key=lambda x: x[0]); return candidates[0][1]
        return None

    def build_border_graph(self, borders: Dict) -> nx.Graph:
        print("="*70); print("STEP 5: BUILD BORDER GRAPH"); print("="*70)
        G = nx.Graph()
        h_lines, v_lines = borders['horizontal'], borders['vertical']
        for line in h_lines+v_lines:
            G.add_node(line['line_id'], type=line['type'], position=line['position'], start=line['start'],
                       end=line['end'], length=line['length'], node_type='border')
        for h in h_lines:
            for v in v_lines:
                if h['start']<=v['position']<=h['end'] and v['start']<=h['position']<=v['end']:
                    G.add_edge(h['line_id'], v['line_id'], intersection_point=(v['position'], h['position']),
                               edge_type='intersection')
        print(f"✓ Border Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        return G

    def build_word_graph(self, words: List[Dict], k: int = 4) -> nx.DiGraph:
        print("="*70); print("STEP 6: BUILD WORD GRAPH"); print("="*70)
        G = nx.DiGraph()
        for w in words: G.add_node(w['word_id'], text=w['text'], bbox=w['bbox'], center=w['center'],
                                   confidence=w['confidence'], node_type='word')
        for i, w1 in enumerate(words):
            distances=[]
            for j, w2 in enumerate(words):
                if i!=j: dist=np.sqrt((w2['center'][0]-w1['center'][0])**2 + (w2['center'][1]-w1['center'][1])**2); distances.append((j,dist,w2))
            distances.sort(key=lambda x: x[1])
            for j, dist, w2 in distances[:k]:
                dx=w2['center'][0]-w1['center'][0]; dy=w2['center'][1]-w1['center'][1]
                if abs(dx)>abs(dy): direction='horizontal'; edge_type='right' if dx>0 else 'left'
                else: direction='vertical'; edge_type='down' if dy>0 else 'up'
                G.add_edge(w1['word_id'], w2['word_id'], direction=direction, edge_type=edge_type, distance=dist)
        print(f"✓ Word Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        return G

    def infer_cells(self, mappings: List[Dict], debug_prefix: str = 'debug') -> List[Dict]:
        print("="*70); print("STEP 7: CELL INFERENCE"); print("="*70)
        cell_groups = defaultdict(list)
        for m in mappings:
            if 'cell_defined_by' in m: cell_groups[tuple(sorted(m['cell_defined_by']))].append(m)
        cells=[]
        for cell_id, (signature, word_mappings) in enumerate(cell_groups.items()):
            top_pos=min([m['borders']['top']['position'] for m in word_mappings if 'top' in m['borders']], default=0)
            bottom_pos=max([m['borders']['bottom']['position'] for m in word_mappings if 'bottom' in m['borders']], default=0)
            left_pos=min([m['borders']['left']['position'] for m in word_mappings if 'left' in m['borders']], default=0)
            right_pos=max([m['borders']['right']['position'] for m in word_mappings if 'right' in m['borders']], default=0)
            cells.append({'cell_id': f"cell_{cell_id}", 'border_signature': signature,
                          'bbox':[int(left_pos), int(top_pos), int(right_pos), int(bottom_pos)],
                          'center':[(left_pos+right_pos)/2, (top_pos+bottom_pos)/2],
                          'word_ids':[m['word_id'] for m in word_mappings],
                          'text':' '.join([m['word_text'] for m in word_mappings])})
        print(f"✓ Inferred {len(cells)} cells\n")
        return cells

    def visualize_all_combined(self, image: np.ndarray, borders: Dict, cells: List[Dict], words: List[Dict],
                               output_path: str = 'viz_all.png'):
        print("="*70); print("FINAL VISUALIZATION"); print("="*70)
        img = Image.fromarray(image); draw=ImageDraw.Draw(img)
        for h in borders['horizontal']: y=int(h['position']); draw.line([(int(h['start']),y),(int(h['end']),y)], fill=(255,0,0), width=2)
        for v in borders['vertical']: x=int(v['position']); draw.line([(x,int(v['start'])),(x,int(v['end']))], fill=(0,0,255), width=2)
        for c in cells: x1,y1,x2,y2=c['bbox']; draw.rectangle([x1,y1,x2,y2], outline=(255,255,0), width=3)
        for w in words: x1,y1,x2,y2=w['bbox']; draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=1)
        img.save(output_path)
        print(f"✓ Saved combined visualization: {output_path}\n")

    def export_json(self, cells: List[Dict], words: List[Dict], output_path: str = 'table_data.json'):
        data={'cells':cells,'words':words}
        with open(output_path,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=4)
        print(f"✓ Exported table data JSON: {output_path}\n")

    def process_table(self, image: Image.Image, debug_prefix: str = 'debug'):
        tables = self.detect_tables(image, debug_prefix=debug_prefix)
        if not tables: print("⚠ No tables detected."); return

        image_np = np.array(image)
        for idx, table in enumerate(tables):
            x1, y1, x2, y2 = table['bbox']
            table_img = image_np[y1:y2, x1:x2]
            table_debug_prefix = f"{debug_prefix}_table{idx+1}"

            borders = self.extract_borders(table_img, debug_prefix=table_debug_prefix)
            words = self.extract_words(table_img, debug_prefix=table_debug_prefix)
            mappings = self.map_words_to_borders(words, borders)
            cells = self.infer_cells(mappings, debug_prefix=table_debug_prefix)

            output_viz_path = os.path.join('out', f"{table_debug_prefix}_combined.png")
            os.makedirs(os.path.dirname(output_viz_path), exist_ok=True)
            self.visualize_all_combined(table_img, borders, cells, words, output_viz_path)

            output_json_path = os.path.join('out', f"{table_debug_prefix}_data.json")
            self.export_json(cells, words, output_json_path)
