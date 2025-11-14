# FairScan Dataset

This repository contains a small, manually annotated dataset for training and evaluating
the [segmentation model](https://github.com/pynicolas/fairscan-segmentation-model) used by
[FairScan](https://github.com/pynicolas/FairScan) to detect documents.

## Overview

- **Contents**: Images of documents (photos) and corresponding binary masks
- **Format**: JPEG images + PNG masks (1 channel)
- **Annotations**: Created manually using [LabelMe](https://github.com/wkentaro/labelme)
- **License**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Structure

The dataset is available via the [Releases](https://github.com/pynicolas/fairscan-dataset/releases) section of this repository. It is structured as follows:

```
.
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── masks/
│       ├── img001.png
│       └── ...
└──  val/
    ├── images/
    └── masks/
```

## Example
Here's an example of an image and the associated mask:

![image](examples/images/image1.jpg)
![mask](examples/masks/image1.png)

## License

The dataset is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:

- **Attribution** — You must give appropriate credit.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license.

See the full [license text here](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


---

Feel free to open an issue if you notice errors in the annotations or would like to contribute improvements.
