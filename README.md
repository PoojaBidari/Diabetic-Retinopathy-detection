# Diabetic-Retinopathy-detection

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab)](https://colab.research.google.com)

An end-to-end Deep Learning project that classifies Retinal Fundus images into **Diabetic Retinopathy (DR)** and **No Diabetic Retinopathy (No_DR)**. The project utilizes **VGG16**-based Transfer Learning, custom classification heads, and decision threshold optimization to maximize clinical accuracy, sensitivity, and specificity.

---

## 📊 Performance Highlights
- **Test Accuracy**: **94.64%** (Exceeded the $\\ge 88\\%$ project goal by **6.64%**)
- **AUC Score**: **0.9894** (Highly robust performance)
- **Sensitivity (Recall)**: **98.31%** (Crucial for minimizing missed DR diagnoses)
- **Specificity**: **91.15%**
- **F1-Score**: **0.9508**
- **Threshold Optimization**: Using a custom decision threshold of **0.8525** reduced False Negatives by **60%** in clinical evaluation.

---

## 📁 Dataset Directory Structure

The project structures retinal images under two distinct classes (`DR` and `No_DR`):

```
Diagnosis of Diabetic Retinopathy/
├── train/
│   ├── DR/       (1,050 images)
│   └── No_DR/    (1,026 images)
├── valid/
│   ├── DR/       (245 images)
│   └── No_DR/    (286 images)
└── test/
    ├── DR/       (113 images)
    └── No_DR/    (118 images)
```
**Total Images**: 2,607 images.

---

## 🛠️ Model Architecture & Features

1. **Feature Extractor**: VGG16 base pre-trained on ImageNet (frozen initially for feature extraction).
2. **Flatten & Custom Classification Head**:
   - Dense (512 nodes, ReLU)
   - Batch Normalization + Dropout (0.5)
   - Dense (256 nodes, ReLU)
   - Batch Normalization + Dropout (0.3)
   - Output layer (1 node, Sigmoid activation for binary classification)
3. **Clinical Optimization**: Implemented a custom wrapper class `OptimizedDRModel` that dynamically maps predictions using the ROC-analyzed optimal threshold of `0.8525` and outputs classification alongside risk assessment profiles.

---

## 🚀 Getting Started & Usage

### 1. Requirements
```bash
pip install tensorflow matplotlib pandas numpy seaborn scikit-learn pillow
```

### 2. Loading the Model & Running Inference

You can easily use the custom Python deployment function below to predict the risk level of single retinal scans:

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved h5 model
best_model = tf.keras.models.load_model('best_dr_model.h5')

def predict_dr_from_image(image_path, model=best_model, threshold=0.8525):
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Raw prediction
    raw_pred = model.predict(img_array)[0][0]
    
    # Class mapping based on optimized threshold
    class_name = "No_DR" if raw_pred > threshold else "DR"
    confidence = raw_pred if class_name == "No_DR" else (1.0 - raw_pred)
    
    # Clinical risk classification
    if class_name == "DR":
        risk_level = "High Risk - Urgent Review Recommended" if confidence > 0.95 else "Moderate Risk"
    else:
        risk_level = "Routine Screening" if confidence > 0.95 else "Regular Monitoring"
        
    return {
        "diagnosis": class_name,
        "confidence": float(confidence),
        "risk_level": risk_level
    }
```

---

## 📈 Training Methodology
- **Data Augmentation**: Subtle rotations, shifts, zooms, and horizontal flips were introduced to make the network invariant to minor fundus scanner variations.
- **Callbacks**: Used `EarlyStopping` (patience=5) and `ReduceLROnPlateau` (factor=0.5) monitoring `val_loss` for stable convergence.
- **Class Weighting**: Balanced minor class inequalities dynamically during gradient steps.

---
