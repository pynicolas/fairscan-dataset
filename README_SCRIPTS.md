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

## Add images
```bash
python resize_images.py
# After having annotated files with labelme
python create_labelme_quads.py # initialize quads that should be checked and fixed manually with labelme
```

## Generate a dataset 

Build a dataset based on `images.csv`. The script avoids having images with the same doc_id in both `train` and `val`.

### For semantic segmentation

```bash
python build_dataset.py
```

### For YOLO

```bash
~/dev/yolo/venv/bin/labelme2yolo --json_dir images_resized --output_format=polygon --val_size 0

python build_dataset.py --segmentation yolo
```
