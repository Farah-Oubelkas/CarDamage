# Car Damage Detection with SwAV + YOLOv8 + DETR

This repository presents a novel deep learning framework that integrates **SwAV-based unsupervised contrastive learning**, **CD-RPN-K region proposal generation**, and a **transformer-based DETR object detector** to identify different types of car damage under uncontrolled conditions.

---

## 🚗 Overview

Car damage detection in real-world environments is challenged by noise, lighting conditions, reflections, and occlusions. This model addresses these problems by combining:

- **Feature encoder (`fθ`)** for deep visual representations.
- **CD-RPN-K** to generate high-quality object proposals from raw input images using a contrastive and image-enhancement approach.
- **SwAV module** to learn robust representations from these proposals via swapped prediction.
- **DETR head** that predicts bounding boxes and damage categories using transformer-based end-to-end detection.

---

## 🧩 Modules

### 1. `FeatureEncoder`
- Based on pretrained backbones (e.g., ResNet50 via `timm`)
- Extracts spatially rich feature maps

### 2. `CD-RPN-K`
- **CDOE**: Applies image transformations and luminance Gaussian filtering
- **Space-K**: Reduces environmental noise like shadows and reflections using learnable filters
- Proposes object-level crops for SwAV and detection

### 3. `SwAVModule`
- Learns self-supervised embeddings from the proposals
- Clusters features to prototypes via contrastive learning

### 4. `DETRHead`
- Predicts bounding boxes and classes using transformer layers
- Implements one-to-one matching between predicted and ground truth boxes using **Hungarian Matching** via a cost matrix

---

## 🛠️ Installation

```bash
git clone https://github.com/Farah-Oubelkas/car-damage-detection
cd car-damage-detection
pip install -r requirements.txt

# python train.py --data path/to/images --epochs 50 --batch-size 16



# CarDamage Dataset
We are in the process of blurring the photos to prevent any privacy concerns. Once this is done, we will share them here. We appreciate your understanding and support.
