# Credit Scoring Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project to predict credit risk (Good / Bad) using the **German Credit Dataset**. Two classification models are trained and compared:

- **Logistic Regression**
- **Random Forest Classifier**

---

## 📁 Project Structure

```
codealpha_task2_Credit_Scoring_Prediction/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── dataset/
│   └── german_credit_data.csv          # Raw dataset
│
├── source_code/
│   └── Credit_Score_prediction.py      # Main Python script
│
├── report/
│   └── Credit_Scoring_Prediction_Report.pdf
│
├── images/
│   ├── age_distribution.png
│   ├── risk_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_auc_logistic.png
│   ├── roc_auc_random_forest.png
│   └── roc_auc_comparison.png
│
└── outputs/
    ├── logistic_results.txt
    ├── random_forest_results.txt
    ├── classification_report_logistic.txt
    ├── classification_report_random_forest.txt
    └── sample_predictions.txt
```

---

## 📊 Dataset

**German Credit Dataset** — 1000 samples with 10 features describing customer financial attributes.

| Feature | Description |
|---|---|
| Age | Customer age (years) |
| Sex | Gender (male/female) |
| Job | Job type (0-3) |
| Housing | Housing type (own/free/rent) |
| Saving accounts | Savings level |
| Checking account | Checking level |
| Credit amount | Loan amount (DM) |
| Duration | Loan duration (months) |
| Purpose | Loan purpose |
| Risk | **Target** — Good / Bad credit |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/codealpha_task2_Credit_Scoring_Prediction.git
cd codealpha_task2_Credit_Scoring_Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Script

```bash
python source_code/Credit_Score_prediction.py
```

---

## 🤖 Models Used

### Logistic Regression
- `max_iter=5000`, `random_state=42`
- Fast baseline classifier

### Random Forest Classifier
- `n_estimators=100`, `random_state=42`
- Ensemble method for improved accuracy

---

## 📈 Results Summary

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | ~75% | ~78% |
| ROC-AUC | ~0.79 | ~0.81 |

> Exact values are available in `outputs/`.

---

## 🖼️ Visualizations

All plots are saved in the `images/` directory:

- Age distribution histogram
- Risk class distribution (Good vs Bad)
- Correlation heatmap
- Confusion matrices for both models
- ROC-AUC curves and comparison

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Abhishek**  
CodeAlpha Internship — Task 2: Credit Scoring Prediction
