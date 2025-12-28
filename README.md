# ML Pricing Challenge - Hybrid Image & Text Model

A machine learning solution for product price prediction using a hybrid deep learning model that combines image and text features. This project implements a multi-modal neural network architecture leveraging EfficientNetB0 for image feature extraction and TF-IDF for text feature extraction.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Required Files](#required-files)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Directory Structure](#directory-structure)

## 🎯 Overview

This project implements a hybrid neural network model that predicts product prices by analyzing both product images and textual descriptions. The model combines:

- **Image Features**: Extracted using EfficientNetB0 (pretrained on ImageNet)
- **Text Features**: Extracted using TF-IDF vectorization
- **Hybrid Architecture**: Multi-input neural network that fuses both modalities

## ✨ Features

- Multi-modal feature extraction (images + text)
- EfficientNetB0-based image feature extraction
- TF-IDF text vectorization
- Hybrid neural network architecture
- End-to-end training and inference pipeline
- Parallel image processing support

## 📁 Project Structure

```
ml-pricing-challenge/
├── dataset/                  # Training and test datasets
│   ├── train.csv
│   ├── test.csv
│   └── sample_test.csv
├── features/                 # Preprocessed features (training)
│   ├── train_features.csv
│   ├── test_features.csv
│   ├── train_img_ids.csv
│   └── test_img_ids.csv
├── features_combined/        # Combined features directory
│   ├── metadata.csv
│   └── image_ids.csv
├── images_jpg/              # Product images (JPG format)
├── outputs/                 # Model predictions
│   └── submission.csv
├── src/                     # Source code
│   ├── download_images.py
│   ├── image_preprocessing.py
│   ├── text_preprocessing.py
│   ├── train_model.py
│   ├── test.py
│   └── test_text_image_preprocess.py
└── README.md
```

## 🔧 Prerequisites

- Python 3.7+
- TensorFlow 2.x
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ml-pricing-challenge
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow pandas numpy scikit-learn joblib tqdm pillow requests
   ```

   Or install from requirements file (if provided):
   ```bash
   pip install -r requirements.txt
   ```

## 📥 Required Files

**⚠️ Important:** The following preprocessed model files and features are required to run inference. These files are large and should be downloaded separately.

### Download Required Files

Please download the following files from the Google Drive link and place them in their respective directories:

**📎 Download Link:** [Google Drive - Model & Feature Files](**YOUR_DRIVE_LINK_HERE**)

### Files to Download:

#### For Training (`features/` directory):
- `hybrid_model_trained.h5` (40.8 MB) - Trained hybrid model weights
- `train_img_features.npy` (751 MB) - Preprocessed training image features
- `X_train_tfidf.pkl` (61.2 MB) - Training text TF-IDF features
- `brand_encoder.pkl` (1 KB) - Brand label encoder

#### For Testing (`features/` directory):
- `test_img_features.npy` (1 MB) - Preprocessed test image features

#### For Combined Features (`features_combined/` directory):
- `image_features.npy` (501 KB) - Combined image features
- `text_tfidf.pkl` (63 KB) - Combined text TF-IDF features
- `tfidf_vectorizer.pkl` (33 KB) - TF-IDF vectorizer model
- `X_test_tfidf.pkl` (61.3 MB) - Test text TF-IDF features (if needed)

### Directory Setup After Download:

```
features/
├── hybrid_model_trained.h5
├── train_img_features.npy
├── test_img_features.npy
├── X_train_tfidf.pkl
├── X_test_tfidf.pkl
├── brand_encoder.pkl
├── train_features.csv
├── test_features.csv
├── train_img_ids.csv
└── test_img_ids.csv

features_combined/
├── image_features.npy
├── text_tfidf.pkl
├── tfidf_vectorizer.pkl
├── metadata.csv
└── image_ids.csv
```

## 🚀 Usage

### 1. Training the Model

To train the hybrid model from scratch:

```bash
python src/train_model.py
```

**Note:** Training requires the preprocessed feature files mentioned above in the `features/` directory.

### 2. Making Predictions

To generate predictions on test data:

```bash
python src/test.py
```

This will:
- Load the trained model
- Load preprocessed image and text features
- Generate price predictions
- Save results to `outputs/submission.csv`

### 3. Preprocessing Steps

If you need to preprocess raw data:

**Text Preprocessing:**
```bash
python src/text_preprocessing.py
```

**Image Preprocessing:**
```bash
python src/image_preprocessing.py
```

**Download Images:**
```bash
python src/download_images.py
```

## 🏗️ Model Architecture

The hybrid model consists of two parallel branches that are concatenated:

```
Input (Image Features) → Dense(512) → Dropout(0.3)
                                              ↓
                                          Concatenate → Dense(256) → Dropout(0.3) → Output (Price)
                                              ↑
Input (Text Features) → Dense(512) → Dropout(0.3)
```

**Key Components:**
- **Image Branch**: Processes 1280-dimensional features from EfficientNetB0
- **Text Branch**: Processes TF-IDF vectorized text features (5000 dimensions)
- **Fusion Layer**: Concatenates both branches and passes through dense layers
- **Output**: Single regression output for price prediction

**Training Configuration:**
- Optimizer: Adam (learning rate: 1e-4)
- Loss: Mean Absolute Error (MAE)
- Metrics: MAE, MSE
- Batch Size: 64
- Epochs: 25 (with early stopping)
- Validation Split: 20%

## 📊 Directory Structure Details

- **`dataset/`**: Contains the raw training and test CSV files
- **`features/`**: Preprocessed features and trained models for training pipeline
- **`features_combined/`**: Combined features for inference/testing
- **`images_jpg/`**: Product images in JPEG format
- **`outputs/`**: Generated predictions and submission files
- **`src/`**: Source code for preprocessing, training, and inference
- **`.venv/`**: Virtual environment (excluded from repository)

## 📝 Notes

- The virtual environment (`.venv/`) is excluded from the repository
- Large model and feature files are hosted separately on Google Drive
- Ensure sufficient disk space (~1.5 GB) for all required files
- For GPU acceleration, ensure CUDA and cuDNN are properly installed

## 🤝 Contributing

This is a competition submission. For questions or issues, please refer to the challenge documentation.


**Last Updated:** October 2025

