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
    COLORS = {
        0: (0, 200, 0),
        1: (0, 0, 200),
        2: (200, 0, 0),
    }

    overlay = image.copy()
    h, w = image.shape[:2]

    for start, end, count in segments:
        if count not in COLORS:
            continue

        if axis == "x":
            cv2.rectangle(overlay, (start, 0), (end, h), COLORS[count], -1)
        else:
            cv2.rectangle(overlay, (0, start), (w, end), COLORS[count], -1)

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


def prune_blocking_segments(segments):
    """
    Apply neighborhood-based pruning rules:
    - 2 -> 1 if neighbor has 1
    - 1 -> 0 if neighbor has 0
    """

    pruned = []
    n = len(segments)

    for i, (start, end, count) in enumerate(segments):
        left = segments[i - 1][2] if i > 0 else None
        right = segments[i + 1][2] if i < n - 1 else None

        new_count = count

        if count == 2 and (left == 1 or right == 1):
            new_count = 1

        elif count == 1 and (left == 0 or right == 0):
            new_count = 0

        pruned.append([start, end, new_count])

    return pruned


def suppress_by_neighbor_context(segments, sentinel=9999):
    """
    Invalidate segments based on neighbor counts.

    Rules:
    - 2 next to 1  → invalidate the 2
    - 1 next to 0  → invalidate the 1

    Invalidation = set count to sentinel (very high number)
    """
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
    """
    Invalidate all segments with count == 2
    by setting them to a high sentinel value.
    """
    result = [seg.copy() for seg in segments]

    for seg in result:
        if seg[2] == 2:
            seg[2] = sentinel

    return result


# -----------------------------
# Execution
# -----------------------------

image_path = "1_table_0.jpg"
doc = detect_text_lines(image_path, "1_table_0_boxes.jpg")

img = cv2.imread(image_path)
h, w = img.shape[:2]

segments = blocking_segments(doc, w, axis="x")
segments = suppress_by_neighbor_context(segments)
segments = remove_all_twos(segments)

x_gaps = find_light_gaps(doc, w, axis="x")
y_gaps = find_light_gaps(doc, h, axis="y")
y_gaps, _ = filter_thin_ranges(y_gaps)

boxed = draw_boxes(image_path, doc)
annotated = draw_blocking_bands(boxed, segments)

annotated = draw_axis_bands(annotated, x_gaps, axis="x")
annotated = draw_axis_bands(annotated, y_gaps, axis="y")

cv2.imwrite("annotated_columns.png", annotated)
plot_segments_with_gradient(segments, w)
