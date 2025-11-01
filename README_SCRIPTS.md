# Tools to build a dataset for FairScan

## Directories
- `images_raw`: raw images captured by a phone
- `images_resized`: 
  - images resized to a reasonable size (e.g. 1024×768)
  - LabelMe JSON files containing annotations
- `image.csv`: database of all dataset images (training + validation + discarded)

Data (images, CSV...) is not committed to the repository.

## Preparing the environment

```bash
python -m venv venv
source venv/bin/activate         # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Generate a dataset for semantic segmentation (fairscan-segmentation-model v1)

```bash
python resize_images.py

# Build dataset based on images.csv (the script avoids having the same doc_id in train and val)
python build_dataset.py --segmentation semantic

```

## Generate a dataset for YOLO

```bash
python resize_images.py

# After having annotated files with labelme:
~/dev/yolo/venv/bin/labelme2yolo --json_dir images_resized --output_format=polygon --val_size 0

# Build dataset based on images.csv (the script avoids having the same doc_id in train and val)
python build_dataset.py --segmentation yolo

```
