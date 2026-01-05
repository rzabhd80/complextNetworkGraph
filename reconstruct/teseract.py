import json
import re

import cv2
import pytesseract
from pytesseract import Output


def detect_text_lines(image_path, output_image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run Tesseract (TSV gives us structure)
    data = pytesseract.image_to_data(
        rgb,
        output_type=Output.DICT,
        config="--psm 6 -c preserve_interword_spaces=1",
    )

    lines = []

    n = len(data["level"])
    for i in range(n):
        # Level 5 = word level
        if data["level"][i] == 5:
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            text = data["text"][i].strip()

            if not text:
                continue

            # Skip text that is only single characters or non-word junk
            if len(text) < 2 or not re.search(r"[a-zA-Z0-9]", text):
                continue

            line_entry = {"text": text, "bbox": [x, y, x + w, y + h]}
            lines.append(line_entry)

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save annotated image
    cv2.imwrite(output_image_path, image)

    # Return JSON-ready structure
    return {"image": image_path, "lines": lines}


if __name__ == "__main__":
    result = detect_text_lines("../out/0_table_0.jpg", "output_with_boxes.jpg")

    with open("lines.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
