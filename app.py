import os

from table_extractor_transformer import HybridTableExtractor

if __name__ == "__main__":
    extractor = HybridTableExtractor(use_fp16=True)

    # Process table and create ALL visualizations automatically
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "dataset", "1.jpg")

    result = extractor.process_table(image_path, output_prefix='output')

    if result:
        # Export JSON
        extractor.export_json(result, 'output.json')

        print("\n" + "=" * 70)
        print("OUTPUT FILES CREATED:")
        print("=" * 70)
        print("📊 output_borders.png  - Borders only (red=horizontal, blue=vertical)")
        print("📊 output_cells.png    - Cells only (colored outlines)")
        print("📊 output_words.png    - Words only (green boxes + red centers)")
        print("📊 output_all.png      - ALL components combined")
        print("📄 output.json         - Full data export")
        print("=" * 70)