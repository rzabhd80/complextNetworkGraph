import csv

import cv2
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from teseract import detect_text_lines


# -----------------------------
# Geometry helpers
# -----------------------------
def shrink_bbox_x(bbox, padding, min_width=5):
    x1, y1, x2, y2 = bbox
    if (x2 - x1) <= 2 * padding + min_width:
        return bbox
    return [x1 + padding, y1, x2 - padding, y2]


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged


def suppress_by_neighbor_context(segments, sentinel=9999):
    if len(segments) < 2:
        return segments
    result = [seg.copy() for seg in segments]
    for i in range(len(segments)):
        start, end, count = segments[i]
        left = segments[i - 1][2] if i > 0 else None
        right = segments[i + 1][2] if i < len(segments) - 1 else None
        neighbors = {left, right}
        if count == 2 and 1 in neighbors:
            result[i][2] = sentinel
        elif count == 1 and 0 in neighbors:
            result[i][2] = sentinel
    return result


def remove_all_twos(segments, sentinel=9999):
    result = [seg.copy() for seg in segments]
    for seg in result:
        if seg[2] == 2:
            seg[2] = sentinel
    return result


# -----------------------------
# Blocking analysis
# -----------------------------
def blocking_segments(data, max_dim, axis="x", x_padding=0):
    events = []
    for line in data["lines"]:
        bbox = line["bbox"]
        if axis == "x":
            bbox = shrink_bbox_x(bbox, x_padding)
            start, end = bbox[0], bbox[2]
        else:
            start, end = bbox[1], bbox[3]
        if start < end:
            events.append((start, +1))
            events.append((end, -1))
    events += [(0, 0), (max_dim, 0)]
    events.sort()
    segments = []
    count = 0
    prev = events[0][0]
    for pos, delta in events:
        if pos > prev:
            segments.append([prev, pos, count])
        count += delta
        prev = pos
    return segments


def find_light_gaps(data, max_dim, axis="x", x_padding=0):
    intervals = []
    for line in data["lines"]:
        bbox = line["bbox"]
        if axis == "x":
            bbox = shrink_bbox_x(bbox, x_padding)
            intervals.append([bbox[0], bbox[2]])
        else:
            intervals.append([bbox[1], bbox[3]])
    merged = merge_intervals(intervals)
    gaps = []
    prev_end = 0
    for start, end in merged:
        if start > prev_end:
            gaps.append([prev_end, start])
        prev_end = max(prev_end, end)
    if prev_end < max_dim:
        gaps.append([prev_end, max_dim])
    return gaps


def filter_thin_ranges(ranges, factor=0.6):
    if len(ranges) == 0:
        return np.array([]), 0
    arr = np.array(ranges, dtype=int)
    lengths = arr[:, 1] - arr[:, 0]
    threshold = lengths.mean() * factor
    return arr[lengths >= threshold], threshold


# -----------------------------
# Drawing helpers
# -----------------------------
def alpha_overlay(base, overlay, alpha):
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def draw_axis_bands(image, ranges, axis="x", color=(0, 0, 255), alpha=0.3):
    overlay = image.copy()
    h, w = image.shape[:2]
    for start, end in ranges:
        if axis == "x":
            cv2.rectangle(overlay, (start, 0), (end, h), color, -1)
        else:
            cv2.rectangle(overlay, (0, start), (w, end), color, -1)
    return alpha_overlay(image, overlay, alpha)


def draw_blocking_bands(image, segments, axis="x", alpha=0.35):
    COLORS = {0: (0, 200, 0), 1: (0, 0, 200), 2: (200, 0, 0)}
    overlay = image.copy()
    h, w = image.shape[:2]
    for start, end, count in segments:
        if count not in COLORS:
            continue
        color = COLORS[count]
        if axis == "x":
            cv2.rectangle(overlay, (start, 0), (end, h), color, -1)
        else:
            cv2.rectangle(overlay, (0, start), (w, end), color, -1)
    return alpha_overlay(image, overlay, alpha)


def draw_boxes(image_path, data):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")
    if "bbox" in data:
        x1, y1, x2, y2 = map(int, data["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for line in data.get("lines", []):
        x1, y1, x2, y2 = map(int, line["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


# -----------------------------
# Plotting
# -----------------------------
def plot_segments_with_gradient(segments, max_dim, max_height=6, axis="x"):
    norm = colors.Normalize(0, max_height)
    cmap = cm.viridis
    plt.figure(figsize=(14, 4))
    for start, end, count in segments:
        count = min(count, max_height)
        plt.bar(
            start,
            count,
            width=end - start,
            align="edge",
            color=cmap(norm(count)),
            edgecolor="none",
        )
    plt.xlim(0, max_dim)
    plt.ylim(0, max_height)
    plt.xlabel(f"{axis.upper()} axis (px)")
    plt.ylabel("Blocking count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Execution & Table Extraction
# -----------------------------
image_path = "1_table_0.jpg"
doc = detect_text_lines(image_path, "1_table_0_boxes.jpg")

img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Cannot read image: {image_path}")
h, w = img.shape[:2]

# Blocking analysis for column separation
segments = blocking_segments(doc, w, axis="x")
segments = suppress_by_neighbor_context(segments)
segments = remove_all_twos(segments)

# Keep only segments with count <= 1 and use as gaps (empty or single-line areas)
cleaned_segments = [seg for seg in segments if seg[2] <= 1]
x_gaps = [seg[:2] for seg in cleaned_segments]  # [[start, end], ...]

# Horizontal gaps (rows)
y_gaps_raw = find_light_gaps(doc, h, axis="y")
y_gaps, _ = filter_thin_ranges(y_gaps_raw)

# Build column bounds (content areas between gaps)
column_bounds = [(0, w)]
if len(x_gaps) > 0:
    bounds = []
    prev = 0
    for start, end in x_gaps:
        if prev < start:
            bounds.append((prev, start))
        prev = end
    if prev < w:
        bounds.append((prev, w))
    column_bounds = bounds

# Build row bounds
row_bounds = [(0, h)]
if y_gaps.size > 0:
    bounds = []
    prev = 0
    for start, end in y_gaps:
        if prev < start:
            bounds.append((prev, start))
        prev = end
    if prev < h:
        bounds.append((prev, h))
    row_bounds = bounds

num_cols = len(column_bounds)
num_rows = len(row_bounds)

print(f"Detected {num_rows} rows and {num_cols} columns")

# Visualization
boxed = draw_boxes(image_path, doc)
annotated = draw_blocking_bands(boxed, segments)
annotated = draw_axis_bands(
    annotated, x_gaps, axis="x", color=(255, 0, 0), alpha=0.3
)  # red = column separators
annotated = draw_axis_bands(
    annotated, y_gaps, axis="y", color=(0, 255, 0), alpha=0.3
)  # green = row separators
cv2.imwrite("annotated_columns.png", annotated)
plot_segments_with_gradient(segments, w)


# -----------------------------
# Improved Table Extraction: Strict Top-to-Bottom, Left-to-Right
# -----------------------------
def find_column(bbox_x1, bbox_x2):
    center_x = (bbox_x1 + bbox_x2) / 2
    for idx, (left, right) in enumerate(column_bounds):
        if left <= center_x < right:
            return idx
    # Fallback: max overlap
    max_overlap = 0
    best_idx = 0
    for idx, (left, right) in enumerate(column_bounds):
        overlap = max(0, min(bbox_x2, right) - max(bbox_x1, left))
        if overlap > max_overlap:
            max_overlap = overlap
            best_idx = idx
    return best_idx


def find_row(bbox_y1, bbox_y2):
    center_y = (bbox_y1 + bbox_y2) / 2
    for idx, (top, bottom) in enumerate(row_bounds):
        if top <= center_y < bottom:
            return idx
    # Fallback: max overlap
    max_overlap = 0
    best_idx = 0
    for idx, (top, bottom) in enumerate(row_bounds):
        overlap = max(0, min(bbox_y2, bottom) - max(bbox_y1, top))
        if overlap > max_overlap:
            max_overlap = overlap
            best_idx = idx
    return best_idx


# Initialize grid
grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]

# Collect all text lines with position info
lines_with_pos = []
for line in doc["lines"]:
    if "text" not in line or not line["text"].strip():
        continue
    x1, y1, x2, y2 = line["bbox"]
    text = line["text"].strip()

    row_idx = find_row(y1, y2)
    col_idx = find_column(x1, x2)

    # Use top-left corner for more stable sorting within cell
    lines_with_pos.append((row_idx, col_idx, y1, x1, text))

# Sort strictly:
# 1. By row index (top to bottom)
# 2. By column index (left to right)
# 3. By y1 (top to bottom within cell)
# 4. By x1 (left to right within cell)
lines_with_pos.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

# Fill grid in correct reading order
for row_idx, col_idx, _, _, text in lines_with_pos:
    if grid[row_idx][col_idx]:
        grid[row_idx][col_idx] += " " + text
    else:
        grid[row_idx][col_idx] = text

# Write TSV
tsv_path = "extracted_table.tsv"
with open(tsv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    for row in grid:
        writer.writerow(row)

print(
    f"Table successfully extracted to '{tsv_path}' with proper top-to-bottom, left-to-right ordering."
)
