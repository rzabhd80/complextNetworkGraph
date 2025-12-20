import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def scan_document(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None, None

    orig = image.copy()
    ratio = image.shape[0] / 500.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Edge detection
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)  # Slightly lower thresholds for more edges

    # Larger closing to connect edges, especially useful for clipped cases
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find only external contours (better for document outline)
    cnts, _ = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) == 0:
        print("Warning: No contours found.")
        return resized, gray

    # Largest external contour
    doc_cnt = max(cnts, key=cv2.contourArea)

    peri = cv2.arcLength(doc_cnt, True)
    # More lenient approximation to encourage 4 points
    approx = cv2.approxPolyDP(doc_cnt, 0.05 * peri, True)  # Increased from 0.02 to 0.05

    # Check if we got a quadrilateral and it's not too large (not the full image)
    image_area = resized.shape[0] * resized.shape[1]
    contour_area = cv2.contourArea(doc_cnt)

    if len(approx) == 4 and contour_area < image_area * 0.95:  # Valid doc outline
        # Scale back to original
        rect = order_points(approx.reshape(4, 2) * ratio)

        # Compute dimensions
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

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
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        # Enhance
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
        )

        return warped, enhanced

    else:
        # Fallback for clipped/full-frame cases: mild crop + enhance whole image
        print("Document border clipped or fills frame – applying flat scan.")
        h, w = orig.shape[:2]
        crop = orig[
            int(h * 0.02) : int(h * 0.98), int(w * 0.02) : int(w * 0.98)
        ]  # 2% crop to remove any border artifacts

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.adaptiveThreshold(
            crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
        )
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return crop, enhanced_color  # Return cropped color and BW


# Usage remains the same
res_color, res_bw = scan_document("receipt.jpg")
if res_color is not None:
    cv2.imwrite("scanned_color.jpg", res_color)
    cv2.imwrite("scanned_bw.jpg", res_bw)
    print("Success! Images saved.")
else:
    print("Failed to process.")
