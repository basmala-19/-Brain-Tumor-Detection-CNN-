# 🧠 Brain Tumor Detection (Binary Classification)

## 📌 Project Overview  
This project builds a **Convolutional Neural Network (CNN)** model to detect whether a brain MRI image contains a **tumor** or **no tumor**.  
The dataset is taken from [Kaggle – Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).  

The task is simplified into a **binary classification**:  
- **0 → No Tumor**  
- **1 → Tumor (Glioma, Meningioma, Pituitary)**  

---
```
## 📂 Dataset Structure  
The dataset is divided into **Training** and **Testing** sets with 4 categories:  
```
📦 Brain-Tumor-MRI-Dataset
 ┣ 📂 Training
 ┃ ┣ 📂 glioma
 ┃ ┣ 📂 meningioma
 ┃ ┣ 📂 pituitary
 ┃ ┗ 📂 notumor
 ┣ 📂 Testing
 ┃ ┣ 📂 glioma
 ┃ ┣ 📂 meningioma
 ┃ ┣ 📂 pituitary
 ┃ ┗ 📂 notumor
```

✅ In this project, the three tumor classes (**glioma, meningioma, pituitary**) are merged into a single **Tumor class**.

---

## ⚙️ Steps Implemented  

1. **Data Loading & Preprocessing**  
   - Images loaded from folders.  
   - Resized to `64x64` pixels.  
   - Normalized using `keras.utils.normalize`.  
   - Labels assigned: `0 (No Tumor)` and `1 (Tumor)`.  
   - Data shuffled and split into train/test sets.  

2. **Model Architecture (CNN)**  
   - **Conv2D + ReLU + MaxPooling** (3 layers).  
   - **Flatten + Dense(128, relu)**.  
   - **Dropout (0.5)** to reduce overfitting.  
   - **Output Dense(1, sigmoid)** for binary classification.  

3. **Training**  
   - Optimizer: `Adam`  
   - Loss: `Binary Crossentropy`  
   - Epochs: `15`  
   - Batch size: `32`  

4. **Evaluation**  
   - Confusion Matrix.  
   - Classification Report (Precision, Recall, F1-score).  
   - ROC Curve + AUC.  
   - Training/Validation Accuracy & Loss curves.  

---

## 📊 Results  

- **Training Accuracy**: ~99%  
- **Validation Accuracy**: ~99%  
- **No Tumor → Precision**: 0.98 | Recall: 1.00 | F1: 0.99
- **Tumor → Precision**: 1.00 | Recall: 0.99 | F1: 1.00
- **AUC (ROC Curve)** ≈ 0.99  

✅ The model performs extremely well, showing strong generalization without major signs of overfitting.  

---

## 🚀 Deployment  

A **Streamlit App** was built to allow users to upload MRI images and get predictions:  

- Upload an MRI image (`jpg, jpeg, png`).  
- The model classifies it as **"Tumor"** or **"No Tumor"**.  

Run locally:  
```bash
streamlit run app.py

