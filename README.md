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

## 🛠️ Installation

```bash
git clone https://github.com/Farah-Oubelkas/car-damage-detection
cd car-damage-detection
pip install -r requirements.txt

# python train.py --data path/to/images --epochs 50 --batch-size 16



## CarDamage Dataset
We are in the process of blurring the photos to prevent any privacy concerns. Once this is done, we will share them here. We appreciate your understanding and support.
