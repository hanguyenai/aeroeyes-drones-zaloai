# AeroEyes Drone Detection & Tracking System (Zalo AI Challenge 2025)

The pipeline includes three main stages: Detection → Refinement → Tracking.

## 1. System Overview

### 1.1 Detection

-   Apply **CLAHE** and **horizontal-flip TTA**.
-   Run ensemble of: YOLOv8s-p2, YOLOv11s-p2 and D-FINE (DETR-like)
-   Merge outputs using **Weighted Boxes Fusion (WBF)**.

### 1.2 Refinement

-   Filter boxes by **min/max area**.
-   Extract features with **MobileNetV3-Small**.
-   Compute cosine similarity with reference images in `box_grabcut/`.
-   Select the box with the highest combined score.

### 1.3 Tracking

-   Track the final selected box using an **Kalman Filter**.

## 2. Environment Setup

### 2.1 Create Python 3.10.19 Virtual Environment

From the project root:

```bash
# Check you have Python 3.10.19 available
python3 --version     # or python --version

# Create virtual environment
python3.10 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 2.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Dataset Preparation

- Convert raw data to YOLO format using `create_dataset.py` with the following directory structure:
```
data/
└── yolo_dataset/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
├── labels/
│ ├── train/
│ ├── val/
│ └── test/
└── dataset.yaml
```
- The D-FINE model requires COCO-style annotations. Use the following directory structure:
```
data/
└── coco/
├── images/
│ ├── train/
│ └── val/
└── annotations/
├── instances_train.json
└── instances_val.json
```
### 3.3 Reference Image Processing

- Use `reference_image_processor.py` to generate cleaned reference crops (e.g., using GrabCut).

### 3.3 Training

- Training scripts and experiments are organized as Jupyter notebooks in the `/notebooks` directory.
- For D-FINE training, navigate to the `D-FINE` directory (`cd D-FINE`).

### 3.4 Inference
Run inference using:
```bash
python predict.py
```
