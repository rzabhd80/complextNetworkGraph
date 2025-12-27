import os
from pathlib import Path

base_dir = Path("./dataset")


def get_datasets(path=base_dir):
    return_dataset = []
    img_folder = path

    # Search for any image file
    for img_path in img_folder.glob("*.jpg"):
        image = img_folder / f"{img_path.stem}.jpg"

        if image.exists():
            return_dataset.append(image)
        else:
            print(f"Missing mask for: {img_path.name}")
    return return_dataset
