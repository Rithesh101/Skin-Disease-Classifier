# 🩺 Skin Disease Classification using Custom CNN  

A deep learning project that classifies skin diseases using a **custom Convolutional Neural Network (CNN)** trained on the **HAM10000 dataset**. The model achieves **98.96% test accuracy**, outperforming traditional baselines.  

---

## 📌 Features  
- Custom CNN architecture designed for efficiency and accuracy  
- Handles **7 different skin disease classes**  
- High accuracy on unseen test data (generalizes well)  
- Data preprocessing, augmentation, and visualization included  
- Evaluation with **confusion matrix, classification report, and accuracy/loss plots**  

---

## 📊 Dataset: HAM10000  
The **HAM10000 ("Human Against Machine with 10000 training images") dataset** is a large collection of **dermoscopic images of pigmented skin lesions**.  
It includes **7 classes**:  

1. **Melanocytic nevi (nv)**  
2. **Melanoma (mel)**  
3. **Benign keratosis-like lesions (bkl)**  
4. **Basal cell carcinoma (bcc)**  
5. **Actinic keratoses (akiec)**  
6. **Vascular lesions (vasc)**  
7. **Dermatofibroma (df)**  

📂 Dataset source: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  

---

## 🛠️ Tech Stack  
- **Python** 🐍  
- **TensorFlow / Keras** for deep learning  
- **NumPy, Pandas** for data handling  
- **Matplotlib, Seaborn** for visualization  

---

## ⚙️ Model Architecture  
- Input: `224x224x3` dermoscopy images  
- Layers:  
  - Convolutional + ReLU + MaxPooling layers  
  - Dropout & Batch Normalization (for regularization)  
  - Fully Connected (Dense) layers  
- Output: **7-class Softmax**  

---

## 📈 Results  

- **Training Accuracy**: 96.25%  
- **Validation Accuracy**: 98.74%  
- **Test Accuracy**: 98.96%  

### 🔹 Performance Metrics  
- Precision, Recall, F1-score per class  
- Confusion Matrix to visualize misclassifications  
- Accuracy/Loss curves for training & validation  

---

## 📷 Sample Visualizations  

- **Accuracy vs Validation Accuracy**  
  ![Accuracy Plot](images/accuracy.png)  

- **Loss vs Validation Loss**  
  ![Loss Plot](images/loss.png)  

- **Confusion Matrix**  
  ![Confusion Matrix](images/confusion_matrix.png)  

- **ROC Curve per Class**  
  ![ROC Curve](images/roc.png)  

*(Place your saved plots inside an `images/` folder in the repo and they will render here)*  

---

## 🚀 How to Run  

1. Clone the repository  
   ```bash
   git clone https://github.com/<your-username>/skin-disease-classification.git
   cd skin-disease-classification

