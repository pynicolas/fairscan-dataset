import json
import cv2
import numpy as np
from pathlib import Path
from shutil import copy2

# --- Configuration ---
SRC_DIR = Path("segmentation")
DST_DIR = Path("quads")
DST_DIR.mkdir(exist_ok=True)

def polygon_to_quad(points):
    """Approximate a polygon to 4 points using OpenCV."""
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1, 2).tolist()

def make_labelme_quad_json(img_filename, img_shape, quad_points):
    """Create a minimal Labelme JSON with one quadrilateral."""
    h, w = img_shape[:2]
    data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [
            {
                "label": "quad",
                "points": quad_points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    return data

# --- Main processing ---
for json_path in sorted(SRC_DIR.glob("*.json")):
    dst_json_path = DST_DIR / json_path.name
    if dst_json_path.exists():
        # don't overwrite existing quad
        continue
    
    img_path = SRC_DIR / json_path.with_suffix(".jpg").name
    if not img_path.exists():
        print(f"⚠️ Missing image for {json_path.name}")
        continue

    # Copy image
    dst_img_path = DST_DIR / img_path.name
    copy2(img_path, dst_img_path)

    # Load segmentation polygons
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("shapes"):
        print(f"⚠️ No shapes in {json_path.name}")
        continue

    first_shape = data["shapes"][0]
    poly_points = np.array(first_shape["points"], dtype=np.float32)

    # Compute quad
    quad_points = polygon_to_quad(poly_points)
    if len(quad_points) != 4:
        print(f"⚠️ Failed to find quad for {img_path}")
        continue

    # Create new JSON
    img = cv2.imread(str(img_path))
    quad_json = make_labelme_quad_json(img_path.name, img.shape, quad_points)
    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(quad_json, f, ensure_ascii=False, indent=2)

    print(f"✅ Created {dst_json_path.name}")

print(f"\n✅ Quadrilateral dataset created in '{DST_DIR}'")
