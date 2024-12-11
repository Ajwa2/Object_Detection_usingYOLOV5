from PIL import Image, ImageDraw, ImageFont
import os

# Load text lines from the file
with open('geez_text_lines_3.txt', 'r', encoding='utf-8') as f:
    geez_text_lines = f.read().splitlines()

# Create a directory for the images
output_dir = 'generated_images/three'
os.makedirs(output_dir, exist_ok=True)

# Define font and size
font_path = 'AbyssinicaSIL-Regular.ttf'  # Replace with the path to a Geez font
font_size = 32
font = ImageFont.truetype(font_path, font_size)
padding = 10  # Add padding to ensure characters fit within the image

# Generate image for each line
for i, line in enumerate(geez_text_lines):
    # Get bounding box of the text
    left, top, right, bottom = font.getbbox(line)
    width, height = right - left, bottom - top
    
    # Add padding to height
    height += padding
    
    # Create a new blank image
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw the text on the image
    draw.text((0, 0), line, font=font, fill='black')

    # Save the image
    image_path = os.path.join(output_dir, f'line_{i+1}.png')
    image.save(image_path)
    print(f"Saved: {image_path}")
