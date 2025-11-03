# Tools to build a dataset for FairScan

## Directories
- `images_raw`: raw images captured by a phone
- `segmentation`: 
  - images resized to a reasonable size (e.g. 1024×768)
  - LabelMe JSON files containing annotations for segmentation (1 polygon for each document in each image)
- `quads`
  - images resized to a reasonable size (e.g. 1024×768)
  - LabelMe JSON files containing annotations for quadrilaterals (1 quad for each image)
- `image.csv`: database of all dataset images (training + validation + discarded)

Data (images, CSV...) is not committed to the git repository.

## Preparing the environment

```bash
python -m venv venv
source venv/bin/activate         # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Adding images
Resize/copy images from `images_raw` to `segmentation`:
```bash
python add_images_for_segmentation.py
```
Then use Labelme to create segmentation polygons for all documents in each image of `segmentation`.

Then initialize quads based on segmentation (the script approximates the contour as a quadrilateral):
```bash
python add_images_for_quads.py
```
Quads should be reviewed and adjusted manually with labelme.

## Generating a dataset 

Build a dataset based on `images.csv`. The script avoids having images with the same doc_id in both `train` and `val`.

### For semantic segmentation

```bash
python build_dataset.py
```

### For YOLO

```bash
~/dev/yolo/venv/bin/labelme2yolo --json_dir segmentation --output_format=polygon --val_size 0

python build_dataset.py --segmentation yolo
```
