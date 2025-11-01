import os
import io
import json
import base64
import shutil
import argparse
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import zipfile

# --- CONFIG ---
CSV_PATH = "images.csv"
YOLO_IMAGES_DIR = "images_resized/YOLODataset/images/train"
YOLO_LABELS_DIR = "images_resized/YOLODataset/labels/train"
YAML_TEMPLATE = "yolo-dataset-template.yaml"
LABELME_JSON_DIR = "images_resized"
VAL_RATIO = 0.2
RANDOM_SEED = 43


def load_csv():
    df = pd.read_csv(CSV_PATH)
    df["has_doc"] = df["has_doc"].fillna(True)
    df["discarded"] = df["discarded"].fillna(False)
    df = df[df["has_doc"] == True]
    df = df[df["discarded"] == False]
    return df


def split_by_doc(df):
    doc_ids = df["doc_id"].unique()
    train_docs, val_docs = train_test_split(doc_ids, test_size=VAL_RATIO, random_state=RANDOM_SEED)
    df["split"] = df["doc_id"].apply(lambda d: "train" if d in train_docs else "val")
    return df


def build_yolo_dataset(df, output_dir):
    # Map original names to YOLO-renamed files
    yolo_images = [f for f in os.listdir(YOLO_IMAGES_DIR) if f.lower().endswith(".jpg")]
    mapping = {}
    for f in yolo_images:
        prefix = "_".join(f.split("_")[0:3]) + ".jpg"
        if prefix not in mapping:
            mapping[prefix] = f

    df = df[df["img_name"].isin(mapping.keys())]

    # Prepare folders
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Copy files
    for _, row in df.iterrows():
        img_name = mapping[row["img_name"]]
        split = row["split"]
        src_img = os.path.join(YOLO_IMAGES_DIR, img_name)
        src_lbl = os.path.join(YOLO_LABELS_DIR, img_name.replace(".jpg", ".txt"))
        dst_img = os.path.join(output_dir, "images", split, img_name)
        dst_lbl = os.path.join(output_dir, "labels", split, img_name.replace(".jpg", ".txt"))
        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

    shutil.copy2(YAML_TEMPLATE, os.path.join(output_dir, "dataset.yaml"))
    print(f"✅ YOLO dataset built in '{output_dir}'")
    print(f"Train: {len(df[df.split=='train'])}, Val: {len(df[df.split=='val'])}")


def draw_mask_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    image_data = base64.b64decode(data["imageData"])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if len(data["shapes"]) > 0:
        points = data["shapes"][0]["points"]
        draw.polygon(points, outline=255, fill=255)
    else:
        raise f"No polygon for {json_path}"

    binary_mask = mask.point(lambda p: 255 if p > 0 else 0).convert("1")
    return binary_mask


def build_semantic_segmentation_dataset(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

    for _, row in df.iterrows():
        img_name = row["img_name"]
        split = row["split"]

        img_src = os.path.join("images_resized", img_name)
        json_src = os.path.join(LABELME_JSON_DIR, img_name.replace(".jpg", ".json"))

        if not os.path.exists(img_src):
            print(f"⚠️ Missing file: {img_src}")
            continue
        if not os.path.exists(json_src):
            print(f"⚠️ Missing file: {json_src}")
            continue

        img_dst = os.path.join(output_dir, split, "images", img_name)
        mask_dst = os.path.join(output_dir, split, "masks", img_name.replace(".jpg", ".png"))

        shutil.copy2(img_src, img_dst)
        mask = draw_mask_from_json(json_src)
        mask.save(mask_dst)

    # --- Create ZIP archive ---
    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                zipf.write(file_path, arcname)

    print(f"✅ Segmentation dataset built in '{output_dir}' and zipped as '{zip_path}'")
    print(f"Train: {len(df[df.split=='train'])}, Val: {len(df[df.split=='val'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset for YOLO or semantic segmentation model")
    parser.add_argument("--segmentation", choices=["semantic", "yolo"], default="semantic")
    args = parser.parse_args()

    df = load_csv()
    df = split_by_doc(df)

    if args.segmentation == "yolo":
        build_yolo_dataset(df, "YOLODataset")
    else:
        build_semantic_segmentation_dataset(df, "fairscan-dataset")
