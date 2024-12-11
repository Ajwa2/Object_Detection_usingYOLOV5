import json
from PIL import ImageFont

# Load text lines from the file
with open('geez_text_lines_3.txt', 'r', encoding='utf-8') as f:
    geez_text_lines = f.read().splitlines()

# Define font and size for annotation
font_path = 'AbyssinicaSIL-Regular.ttf'  # Replace with the path to a Geez font
font_size = 32
font = ImageFont.truetype(font_path, font_size)
padding = 10  # Ensure this matches the padding used during image generation

# Generate annotations for each image
annotations = []
for i, line in enumerate(geez_text_lines):
    left, top, right, bottom = font.getbbox(line)
    width, height = right - left, bottom - top
    
    # Add padding to height
    height += padding
    
    annotation = {
        "image_id": i + 1,
        "file_name": f'line_{i+1}.png',
        "bbox": [0, 0, width, height],
        "attributes": {"Lable": line}
    }
    annotations.append(annotation)

# Save annotations to a JSON file
annotations_file = 'annotations.json'
with open(annotations_file, 'w', encoding='utf-8') as f:
    json.dump({"annotations": annotations}, f, ensure_ascii=False, indent=4)
    print(f"Saved annotations to: {annotations_file}")
