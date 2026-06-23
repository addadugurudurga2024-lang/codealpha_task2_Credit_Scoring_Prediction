import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
# Load Dataset
df = pd.read_csv("D:/german_credit_data.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Handle Missing Values
df['Saving accounts'] = df['Saving accounts'].fillna('Unknown')
df['Checking account'] = df['Checking account'].fillna('Unknown')

# Remove Unnecessary Column
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
# Data Visualization
# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df['Age'], bins=10)
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.title("Age Distribution")
plt.show()

# Risk Distribution
plt.figure(figsize=(6,5))
df['Risk'].value_counts().plot(kind='bar')
plt.xlabel("Credit Risk")
plt.ylabel("Count")
plt.title("Good vs Bad Credit Risk")
plt.show()

# Label Encoding
categorical_columns = [
    'Sex',
    'Housing',
    'Saving accounts',
    'Checking account',
    'Purpose'
]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# Encode Target Variable Separately
le_risk = LabelEncoder()
df['Risk'] = le_risk.fit_transform(df['Risk'])

print("\nRisk Mapping:")
print(le_risk.classes_)

# ['bad' 'good']
# 0 -> Bad
# 1 -> Good

# Correlation Heatmap
correlation = df.corr()

plt.figure(figsize=(12,8))
sn.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Correlation Heatmap")
plt.show()

# Split Dataset

X = df.drop('Risk', axis=1)
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=5000,random_state=42)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:",accuracy_score(y_test, y_pred_log))
print("Precision:",precision_score(y_test, y_pred_log))
print("Recall:",recall_score(y_test, y_pred_log))
print("F1 Score:",f1_score(y_test, y_pred_log))

prob_log = log_model.predict_proba(X_test)[:,1]

print("ROC-AUC:",roc_auc_score(y_test, prob_log))

print(classification_report(y_test,y_pred_log))

# Confusion Matrix
cm_log = confusion_matrix(y_test,y_pred_log)

plt.figure(figsize=(7,5))
sn.heatmap(cm_log,annot=True,fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:",accuracy_score(y_test,y_pred_rf))
print("Precision:",precision_score(y_test,y_pred_rf))
print("Recall:",recall_score(y_test,y_pred_rf))
print("F1 Score:",f1_score(y_test,y_pred_rf))

prob_rf = rf_model.predict_proba(X_test)[:,1]

print("ROC-AUC:",roc_auc_score(y_test,prob_rf))
print(classification_report(y_test,y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test,y_pred_rf)

plt.figure(figsize=(7,5))
sn.heatmap(cm_rf,annot=True,fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ROC Curve Comparison
fpr_log, tpr_log, _ = roc_curve(y_test,prob_log)

fpr_rf, tpr_rf, _ = roc_curve(y_test,prob_rf)
auc_log = roc_auc_score(y_test,prob_log)

auc_rf = roc_auc_score(y_test,prob_rf)
plt.figure(figsize=(8,6))

plt.plot(fpr_log,tpr_log,label=f'Logistic Regression (AUC={auc_log:.2f})')

plt.plot(fpr_rf,tpr_rf,label=f'Random Forest (AUC={auc_rf:.2f})')
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Comparison")
plt.legend()
plt.show()

# Sample Prediction 1
sample1 = pd.DataFrame(
    [[45,1,2,1,2,2,2000,12,5]],
    columns=X.columns
)

print("\nSample Customer 1")
print(sample1)

print("\nLogistic Regression Prediction")
print(log_model.predict(sample1))
print(log_model.predict_proba(sample1))

print("\nRandom Forest Prediction")
print(rf_model.predict(sample1))
print(rf_model.predict_proba(sample1))

prediction_log = log_model.predict(sample1)
prediction_rf = rf_model.predict(sample1)

if prediction_log[0] == 1:
    print("Logistic -> Good Credit Risk")
else:
    print("Logistic -> Bad Credit Risk")

if prediction_rf[0] == 1:
    print("RandomForest -> Good Credit Risk")
else:
    print("RandomForest -> Bad Credit Risk")

# Sample Prediction 2
sample2 = pd.DataFrame(
    [[22,0,0,2,0,0,15000,60,1]],
    columns=X.columns
)

print("\nSample Customer 2")
print(sample2)

print("\nLogistic Regression Prediction")
print(log_model.predict(sample2))
print(log_model.predict_proba(sample2))

print("\nRandom Forest Prediction")
print(rf_model.predict(sample2))
print(rf_model.predict_proba(sample2))

prediction1_log = log_model.predict(sample2)
prediction1_rf = rf_model.predict(sample2)

if prediction1_log[0] == 1:
    print("Logistic -> Good Credit Risk")
else:
    print("Logistic -> Bad Credit Risk")

if prediction1_rf[0] == 1:
    print("RandomForest -> Good Credit Risk")
else:
    print("RandomForest -> Bad Credit Risk")
