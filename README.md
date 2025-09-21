# Emotion Detection Web App (ONNX Runtime)

## Introduction

This project demonstrates a **complete pipeline for emotion detection** using deep learning.  
It consists of two main parts:

1. **Training & Model Preparation**  
   - A Google Colab Notebook (`Emotion_Detection.ipynb`) that explains the full workflow:  
     data preprocessing, model training, evaluation, and ONNX export.  

2. **Web Application**  
   - A simple web-based interface (`app.js, index.html, style.css`) built with **ONNX Web Runtime**.  
   - Users can upload a **headshot photo** and receive predicted **emotions** in real time.  

Together, these components form an end-to-end solution:  
from **training your own emotion detection model** → to **deploying it in the browser**.

---

## Colab Python Notebook (`/Emotion_Detection.ipynb`)

The Colab notebook walks through the **end-to-end process** of building and exporting an emotion detection model.  

### 1. Setup & Dependencies  
- Install required libraries (`torch`, `torchvision`, `onnxruntime`, `scikit-learn`, etc.).  
- Configure Kaggle API for dataset downloads.  

### 2. Dataset Acquisition  
- Download **FER-2013** and **AffectNet** datasets from Kaggle.  
- Extract archives and organize into training/testing directories.  

### 3. Unified Labeling  
- Map fine-grained emotion labels (angry, sad, happy, etc.) → into **3 broad categories**:  
  - `negative`, `neutral`, `positive`.  
- Define consistent class indices for training.  

### 4. Data Preparation  
- Custom PyTorch `Dataset` class for flexible folder-based loading.  
- Preprocessing pipeline:  
  - Convert grayscale → RGB (3 channels).  
  - Resize to **224×224**.  
  - Normalize using ImageNet mean/std.  
- Combine **FER-2013 + AffectNet** into a unified dataset.  
- Split into **train / validation / test** sets.  

### 5. Model Architecture  
- Load **ResNet18** pretrained on ImageNet.  
- Freeze most layers, fine-tune only:  
  - Last convolutional block.  
  - Final fully-connected (FC) layer → replaced with **3 outputs**.  

### 6. Training Pipeline  
- Loss: CrossEntropy.  
- Optimizer: Adam with LR scheduler.  
- Features:  
  - Mixed-precision training (`GradScaler`).  
  - Early stopping (patience = 15).  
  - Checkpoint saving + best-model export.  

### 7. Evaluation  
- Classification metrics with `sklearn`: accuracy, confusion matrix, per-class scores.  
- Visualization:  
  - Raw confusion matrix.  
  - Normalized confusion matrix.  

### 8. Inference Test  
- Upload a headshot image.  
- Apply the same preprocessing pipeline.  
- Predict emotion label using the trained model.  

### 9. ONNX Export  
- Export trained PyTorch model → ONNX format.  
- Define input/output node names (`input`, `output`).  
- Save `Emotion_Model.onnx` for deployment.  
- Verify model input/output shapes in Colab.  

---

## Web-App Overview (`app.js, index.html, style.css`)

The Web-App allows users to **upload a headshot image** and detect the emotion in real-time using the **ONNX model** exported from the notebook.  

### 1. Features
- Upload and preview an image directly in the browser.
- Preprocess images to match training input size (**224×224**) and normalization.
- Run inference locally in the browser using **ONNX Runtime Web (`onnxruntime-web`)**.
- Display predicted emotion: `negative`, `neutral`, or `positive`.
- Show **probabilities** for each class in a readable percentage format.
- Colored result display based on predicted emotion.

### 2. Key Components
1. **`index.html`**  
   - File input for uploading images (hidden input styled via label).  
   - Preview area for the selected image.  
   - Button to run inference.  
   - Output area to display predicted emotion and class probabilities.  

2. **`app.js`**  
   - Handles image preprocessing: resizing, converting to tensor, normalization.  
   - Loads ONNX model using `ort.InferenceSession`.  
   - Runs inference and extracts output probabilities.  
   - Updates DOM with predicted label and percentage probabilities.  

3. **`style.css`**  
   - Styles buttons, preview image, and results.  
   - Provides responsive layout and colored indicators for predicted emotion.  

### 3. How It Works
1. User clicks **Upload** button → selects an image.
2. Image is previewed in the browser.
3. Click **Run** → `app.js` converts image → ONNX-compatible tensor.
4. ONNX Runtime Web evaluates the model → returns predictions.
5. The app displays:
   - Predicted emotion label (highlighted with color)
   - Class probabilities as percentages (`negative: 82%`, `neutral: 10%`, `positive: 8%`)

### 4. Advantages
- **Client-side inference:** No server required. Runs entirely in the browser.  
- **Lightweight:** Works with a small ResNet18 ONNX model.  
- **User-friendly:** Intuitive interface for end users to analyze headshot images instantly.