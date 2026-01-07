import json

import cv2
import pandas as pd
import pytesseract
from pytesseract import Output

GAP_THRESHOLD_RATIO = 0.5


def detect_text_lines(image_path, output_image_path):
    # 1. Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Run Tesseract
    data = pytesseract.image_to_data(
        thresh,
        output_type=Output.DICT,
        config="--psm 12 -l eng -c preserve_interword_spaces=1",
    )
    df = pd.DataFrame(data)

    # Save the raw TSV
    df.to_csv("output.tsv", sep="\t", index=False, encoding="utf-8")

    # 3. Filter for word-level data (level 5) and remove empty text
    df_words = df[(df["level"] == 5) & (df["text"].str.strip() != "")].copy()

    # Sort words by their appearance order on the page
    df_words = df_words.sort_values(
        ["page_num", "block_num", "par_num", "line_num", "word_num"]
    )

    lines = []
    if not df_words.empty:
        # We define a 'group' as words belonging to the same column-cell
        # A new group starts if:
        # 1. The line/paragraph/block changes
        # 2. The horizontal gap to the previous word is too large (Column break)

        # Calculate the gap between the current word and the previous word
        df_words["prev_right"] = df_words["left"].shift() + df_words["width"].shift()
        df_words["gap"] = df_words["left"] - df_words["prev_right"]

        # Group identifier logic:
        # If gap > height * 1.0, it's likely a different column.
        # You can adjust '1.0' to '0.8' if it still merges or '1.5' if it splits too much.
        df_words["new_group"] = (
            (
                (df_words["line_num"] != df_words["line_num"].shift())
                | (df_words["par_num"] != df_words["par_num"].shift())
                | (df_words["block_num"] != df_words["block_num"].shift())
                | (df_words["gap"] > df_words["height"].shift() * GAP_THRESHOLD_RATIO)
            )
            .fillna(True)
            .cumsum()
        )

        # 4. Merge words by the new group ID
        merged = (
            df_words.groupby("new_group")
            .agg(
                {
                    "left": "min",
                    "top": "min",
                    "width": lambda x: (
                        df_words.loc[x.index, "left"] + df_words.loc[x.index, "width"]
                    ).max()
                    - df_words.loc[x.index, "left"].min(),
                    "height": "max",
                    "text": lambda x: " ".join(map(str, x)),
                }
            )
            .reset_index()
        )

        for _, row in merged.iterrows():
            x, y, w, h = (
                int(row["left"]),
                int(row["top"]),
                int(row["width"]),
                int(row["height"]),
            )
            text = row["text"].strip()

            line_entry = {"text": text, "bbox": [x, y, x + w, y + h]}
            lines.append(line_entry)

            # Draw the box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 5. Save annotated image
    cv2.imwrite(output_image_path, image)

    return {"image": image_path, "lines": lines}


if __name__ == "__main__":
    result = detect_text_lines("../out/12_table_0.jpg", "output_with_boxes.jpg")

    with open("lines.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
