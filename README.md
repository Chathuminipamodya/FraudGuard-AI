# Credit Card Fraud Detection 💳🔍

An end-to-end machine learning project to detect fraudulent credit card transactions using advanced classification algorithms.

## 📊 Project Overview

This project implements multiple machine learning models to identify fraudulent transactions in credit card data. The dataset is highly imbalanced, making it a challenging and realistic fraud detection scenario.

## 🎯 Objectives

- Build and compare multiple ML models for fraud detection
- Handle imbalanced dataset effectively
- Achieve high precision and recall for fraud detection
- Create a production-ready model pipeline

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data
├── notebooks/
│   └── exploration.ipynb # EDA and experiments
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── evaluate.py
├── models/               # Saved trained models
├── results/              # Visualizations and metrics
├── requirements.txt
├── main.py
└── README.md
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**: 30 (V1-V28 from PCA, Time, Amount)
- **Target**: Class (0 = Normal, 1 = Fraud)
- **Imbalance**: 0.172% fraud cases

Download the dataset and place `creditcard.csv` in the `data/raw/` folder.

## 🚀 Usage

### Train the model:
```bash
python main.py --train
```

### Make predictions:
```bash
python main.py --predict --input data/test.csv
```

### Run full pipeline:
```bash
python main.py --full-pipeline
```

## 🤖 Models Implemented

1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble method
3. **XGBoost** - Gradient boosting (best performance)
4. **Neural Network** - Deep learning approach

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

*Results will be updated after training*

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **Imbalanced-learn** - Handling imbalanced data
- **Matplotlib/Seaborn** - Visualization

## 📝 Key Features

- ✅ Comprehensive data preprocessing pipeline
- ✅ Multiple ML algorithms comparison
- ✅ Handling imbalanced data (SMOTE, undersampling)
- ✅ Feature importance analysis
- ✅ Detailed model evaluation metrics
- ✅ Visualization of results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset provided by ULB Machine Learning Group
- Kaggle for hosting the dataset
- Open-source community for amazing tools

---

⭐ If you find this project useful, please give it a star!