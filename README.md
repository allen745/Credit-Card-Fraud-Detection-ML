# Credit-Card-Fraud-Detection-ML
# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Overview

This project uses **Machine Learning (Logistic Regression)** to detect fraudulent credit card transactions. Since fraud cases are extremely rare compared to legitimate transactions, the dataset is highly imbalanced. To address this issue, random undersampling is used to create a balanced dataset before training the model.

The model is trained using Scikit-Learn's Logistic Regression algorithm and evaluated using accuracy metrics.

---

## 🚀 Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Handling imbalanced datasets
- Random undersampling technique
- Logistic Regression classification model
- Model performance evaluation
- Fraud transaction prediction

---

## 📊 Dataset Information

The dataset contains anonymized credit card transactions.

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Total Transactions | 284,807 |
| Legitimate Transactions | 284,315 |
| Fraudulent Transactions | 492 |
| Features | 30 |
| Target Column | Class |

### Target Variable

- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**

---

## 🛠️ Tech Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-Learn

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection-ML.git
```

### Navigate to Project Folder

```bash
cd Credit-Card-Fraud-Detection-ML
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection-ML/
│
├── data/
│   └── creditcard.csv
│
├── Credit Card Fraud Detection.py
│
├── README.md
│
└── requirements.txt
```

---

## ⚙️ Project Workflow

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Data Analysis
      ↓
Handling Imbalanced Dataset
      ↓
Train-Test Split
      ↓
Logistic Regression
      ↓
Model Evaluation
      ↓
Fraud Detection Prediction
```

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

- Loaded dataset using Pandas
- Checked dataset information
- Generated statistical summaries
- Verified missing values
- Analyzed transaction distribution
- Balanced the dataset using undersampling

---

## ⚖️ Handling Imbalanced Data

Original Distribution:

| Class | Count |
|---------|---------|
| Legitimate | 284,315 |
| Fraud | 492 |

A sample of 492 legitimate transactions was selected and combined with all fraudulent transactions to create a balanced dataset.

---

## 🤖 Model Training

### Logistic Regression

```python
model = LogisticRegression()
model.fit(x_train, y_train)
```

---

## 📈 Results

### Training Accuracy

```text
99.89%
```

### Testing Accuracy

```text
99.91%
```

### Output

```text
Accuracy on training data: 0.9989791402871552
Accuracy on testing data: 0.999123729407641
```

---

## ⚠️ Model Improvement

During training, Scikit-Learn displayed a convergence warning:

```text
lbfgs failed to converge after 100 iterations
```

Recommended improvement:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
```

For better performance, feature scaling can also be applied using StandardScaler.

---

## 🔮 Future Enhancements

- Random Forest Classifier
- XGBoost Classifier
- SMOTE Oversampling
- Confusion Matrix Visualization
- Precision, Recall & F1-Score
- Streamlit Web Application
- FastAPI Deployment

---

## 👨‍💻 Author

**Allen Christian** | **Patent Holder**

AI & Machine Learning Engineer

---

## ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub.
