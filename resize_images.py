from pathlib import Path
from PIL import Image

input_dir = Path("images_raw")
output_dir = Path("images_resized")
output_dir.mkdir(exist_ok=True)

max_size = 1024  # pixels

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    ratio = max_size / max(width, height)
    new_size = (int(width * ratio), int(height * ratio))
    return image.resize(new_size, Image.LANCZOS)

for img_path in input_dir.glob("*.jpg"):    
    out_path = output_dir / img_path.name
    if out_path.is_file():
        continue
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        resized = resize_image(img, max_size)    
        resized.save(out_path, format="JPEG", quality=95)
        print(f"→ {img_path.name}: {img.size} → {resized.size}")

