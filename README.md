# 🚗 Offroad Image Classification using Transfer Learning (InceptionV3)

> A deep learning pipeline for binary classification of offroad vs. non-offroad images using InceptionV3 with transfer learning on TensorFlow/Keras.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Challenges](#challenges)
- [Conclusion](#conclusion)

---

## 🎯 Problem Statement

The goal of this project is to build an image classification model that can accurately distinguish between offroad and non-offroad terrain images. This binary classification task was developed as part of a hackathon challenge, using real-world offroad segmentation data.

---

## 📦 Dataset

| Split      | Images  | Classes |
|------------|---------|---------|
| Training   | 5,716   | 2       |
| Validation | 2,003   | 2       |

- **Classes:** 2 (binary — offroad / non-offroad)
- **Input Image Size:** 224 × 224 × 3 (RGB)
- **Source:** `Offroad_Segmentation_Training_Dataset`
- **Augmentation applied to training set:**
  - Rescaling (`1/255`)
  - Shear range: `0.2`
  - Zoom range: `0.2`
  - Horizontal flip: `True`

---

## 🧠 Model Architecture

The model uses **InceptionV3** as a frozen feature extractor (pretrained on ImageNet), with a custom classification head:

```
Input (224, 224, 3)
    ↓
InceptionV3 (frozen — 21,802,784 non-trainable params)
    ↓
Flatten
    ↓
Dense(num_classes=2, activation='softmax')
```

- **Base Model:** `InceptionV3` (weights = `imagenet`, `include_top = False`)
- **All base layers frozen:** `layer.trainable = False`
- **Trainable parameters:** 102,402 (classification head only)
- **Total parameters:** 21,905,186

---

## ⚙️ Training Setup

| Parameter         | Value                    |
|-------------------|--------------------------|
| Framework         | TensorFlow 2.20 / Keras  |
| GPU               | NVIDIA Tesla T4 (Google Colab) |
| Optimizer         | Adam                     |
| Loss Function     | Categorical Crossentropy |
| Metrics           | Accuracy, MeanIoU        |
| Epochs            | 10                       |
| Batch Size        | 32                       |
| Image Size        | 224 × 224                |
| Steps per Epoch   | 179                      |
| Validation Steps  | 63                       |

---

## 📊 Results

### Training History (per epoch)

| Epoch | Train Accuracy | Train Loss     | Val Accuracy | Val Loss    |
|-------|---------------|----------------|--------------|-------------|
| 1     | 99.44%        | 0.0290         | 100.00%      | ~0.000002   |
| 2     | 100.00%       | 0.0024         | 100.00%      | 0.0020      |
| 3     | 99.98%        | 0.0027         | 100.00%      | 0.0001      |
| 4     | 100.00%       | 0.0091         | 99.80%       | 0.0196      |
| 5     | 99.95%        | 0.0071         | 100.00%      | 0.0288      |
| 6     | 99.98%        | 0.0153         | 99.45%       | 0.0469      |
| 7     | 99.99%        | 0.0113         | 99.90%       | 0.0204      |
| 8     | 100.00%       | 6.45e-06       | 100.00%      | 0.0023      |
| 9     | 100.00%       | 5.87e-08       | 100.00%      | 0.0023      |
| 10    | 100.00%       | 0.0051         | 99.20%       | 0.1870      |

### ✅ Final Evaluation Metrics

| Metric        | Score   |
|---------------|---------|
| Val Loss      | 0.1080  |
| **Val IoU**   | **0.9817** |
| **Val Dice**  | **0.9908** |
| **Val Accuracy** | **99.20%** |

> **Initial IoU (before training):** 0.3367  
> **Final IoU (after training):** 0.9817 — a significant improvement of ~+0.645

---

## 🗂️ Project Structure

```
├── model_inception.h5          # Saved trained model (HDF5 format)
├── offroad_classification.ipynb # Main Colab notebook
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Mount Google Drive (if using Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Run the notebook
Open `offroad_classification.ipynb` in Google Colab and run all cells sequentially.

### 5. Load saved model for inference
```python
from tensorflow.keras.models import load_model
model = load_model('model_inception.h5')
```

---

## 📦 Requirements

```
tensorflow>=2.20.0
numpy>=1.26.0
matplotlib
h5py>=3.11.0
Pillow
```

> See `requirements.txt` for full dependency list.

---

## ⚠️ Challenges Faced

- **Flattening InceptionV3 output** produces a very large vector (`51,200` dimensions), which is memory-intensive. Using `GlobalAveragePooling2D` would be more efficient.
- **Data generator mismatch** — the test set used for IoU calculation had to be carefully re-initialized before evaluation to avoid label shuffle issues.
- **Model saving warning** — Keras recommends `.keras` format over `.h5`; the project uses `.h5` for compatibility.
- **Recompilation required** after `load_model` to restore compiled metrics (`MeanIoU`).

---

## 🏁 Conclusion

The InceptionV3-based transfer learning model achieved excellent performance on the offroad image classification task:

- **99.20% validation accuracy**
- **0.9817 IoU score**
- **0.9908 Dice coefficient**

The frozen feature extractor strategy proved highly effective, requiring only ~100K trainable parameters while leveraging rich ImageNet representations. The model converged quickly within just a few epochs, demonstrating the power of transfer learning for domain-specific image classification tasks.

---

## 👥 Team

> Add your team name and member names here.

---

## 📄 License

This project is submitted as part of a hackathon. All rights reserved by the respective team.
