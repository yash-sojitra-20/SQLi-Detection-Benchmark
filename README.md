# SQLi-Detection-Benchmark
Benchmarks ML and DL models for SQL injection (SQLi) detection on both imbalanced and balanced datasets. Uses consistent preprocessing and evaluation metrics to identify the most effective models and data balancing techniques.

## Problem Statement

SQL Injection (SQLi) is one of the most common and dangerous web vulnerabilities.  
Attackers can inject malicious SQL code into input fields to steal or manipulate data.  
Detecting these attacks automatically is a challenge, especially because:

- Attackers use tricks like encoding or hidden syntax.
- Real datasets are **imbalanced** : many safe queries, few attacks.

This project compares various ML and DL models to see which works best for this detection task.

---

## Objectives

- Build a full **benchmarking pipeline** for SQLi detection.
- Test multiple ML and DL models under identical conditions.
- Handle **class imbalance** using:
  - Random Under-Sampling (RUS)
  - Random Over-Sampling (ROS)
- Use **TF-IDF (character-level n-grams)** for feature extraction.
- Evaluate models using metrics like Accuracy, Precision, Recall, F1-score, and PR-AUC.
- Ensemble all models and compare the combined performance.

---

## Models Evaluated

### 🔹 Machine Learning Models
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)

### 🔹 Deep Learning Models
- Multi-Layer Perceptron (MLP)  
- MobileBERT (Transformer-based model)

### 🔹 Ensemble Methods
- Hard Voting (Majority Rule)  
- Soft Voting (Average Probabilities)  
- Weighted Soft Voting (Assigns higher weights to stronger models)

---

## Dataset

- **Source:** Kaggle Public Dataset – *SQL Injection Dataset*  
- **Total Samples:** 30,919  
- **Benign Queries:** 19,537  
- **Malicious Queries:** 11,382  
- **Imbalance Ratio:** ≈ 1.72 : 1  

After preprocessing, three versions of the dataset were used:
1. Original (Imbalanced)
2. RUS – Random Under-Sampling
3. ROS – Random Over-Sampling

---

## Methodology

1. **Load and Split Data** – 80% training, 20% testing  
2. **Text Preprocessing** – lowercase, clean spacing  
3. **Feature Extraction** – TF-IDF with character-level n-grams (3–6 range)  
4. **Resampling** – Apply RUS and ROS  
5. **Model Training** – Train all models on each dataset  
6. **Evaluation** – Use consistent test data and metrics  
7. **Ensembling** – Combine all models for Hard, Soft, and Weighted Voting  

---

## Results Summary

| Model | Dataset | Accuracy | F1-Score | Key Notes |
|--------|----------|----------|-----------|-----------|
| Logistic Regression | Original | 99.34% | 99.34% | Strong linear baseline |
| Decision Tree | ROS | 99.53% | 99.53% | Good improvement with ROS |
| **Random Forest** | **ROS** | **99.76%** | **99.76%** | ⭐ Best overall performer |
| SVM | Original | 99.48% | 99.48% | Stable performance |
| KNN | RUS | 93.52% | 93.51% | Weak on high-dimensional data |
| MLP | ROS | 99.55% | 99.55% | Performs very well with ROS |
| MobileBERT | Original | 99.31% | 99.31% | Highly stable across datasets |

### Ensemble Results

| Dataset | Voting Type | Accuracy | PR-AUC |
|----------|--------------|----------|--------|
| Original | Weighted Soft | 99.60% | 0.9991 |
| RUS | Weighted Soft | 99.36% | 0.9994 |
| **ROS** | **Weighted Soft** | **99.64%** | **0.9999** |

✅ **Best Configuration:** Random Forest + ROS  
✅ **Best Ensemble:** Weighted Soft Voting + ROS  

---

## Key Findings

- Random Forest performed the best individually.  
- Oversampling (ROS) improved almost every model.  
- KNN did not perform well due to high-dimensional sparse vectors.  

---

## 💡 Future Improvements

- Try advanced balancing techniques like **SMOTE** and **ADASYN**.  
- Include more transformer models (DistilBERT, RoBERTa).  
- Add more features such as query length, keyword ratio, or embeddings.  
- Build a small **web demo** to show real-time SQLi detection.

---

## 🛠️ Tech Stack

- **Language:** Python 3  
- **Libraries:**  
  - `scikit-learn`, `numpy`, `pandas`, `matplotlib`  
  - `transformers` (for MobileBERT)  

---

## 🧑‍💻 Contributors

| Name                     | Role                                             |
| ------------------------ | ------------------------------------------------ |
| **Yash P. Sojitra**      | Data Preparation, Model Implementation, Experiments |
| **Khushi V. Zalavadiya** | Documentation, Evaluation, Analysis |

---
