# Skin Disease Classification using Custom CNN  

A deep learning project that classifies skin diseases using a **custom Convolutional Neural Network (CNN)** trained on the **HAM10000 dataset**. The model achieves **98.96% test accuracy**, outperforming traditional baselines.  

---

## Features  
- Custom CNN architecture designed for efficiency and accuracy  
- Handles **7 different skin disease classes**  
- High accuracy on unseen test data (generalizes well)  
- Data preprocessing, augmentation, and visualization included  
- Evaluation with **confusion matrix, classification report, and accuracy/loss plots**  

---

## Dataset: HAM10000  
The **HAM10000 ("Human Against Machine with 10000 training images") dataset** is a large collection of **dermoscopic images of pigmented skin lesions**.  
It includes **7 classes**:  

1. **Melanocytic nevi (nv)**  
2. **Melanoma (mel)**  
3. **Benign keratosis-like lesions (bkl)**  
4. **Basal cell carcinoma (bcc)**  
5. **Actinic keratoses (akiec)**  
6. **Vascular lesions (vasc)**  
7. **Dermatofibroma (df)**  

Dataset source: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  

---

## 🛠️ Tech Stack  
- **Python**   
- **TensorFlow / Keras** for deep learning  
- **NumPy, Pandas** for data handling  
- **Matplotlib, Seaborn** for visualization  

---

##  Model Architecture  
- Input: `224x224x3` dermoscopy images  
- Layers:  
  - Convolutional + ReLU + MaxPooling layers  
  - Dropout & Batch Normalization (for regularization)  
  - Fully Connected (Dense) layers  
- Output: **7-class Softmax**  

---

## Results  

- **Training Accuracy**: 96.25%  
- **Validation Accuracy**: 98.74%  
- **Test Accuracy**: 98.96%  

---

## 🚀 How to Run  

1. Clone the repository  
   ```bash
   git clone https://github.com/Rithesh101/skin-disease-classification.git
   cd skin-disease-classification

2. Install dependencies
   ```bash
   pip install -r requirements.txt   
3. Run Flask app
   ```bash
   python app.py
