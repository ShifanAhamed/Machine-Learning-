# File: complaint_resolution_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1. Load CSV Dataset
# ---------------------------
df = pd.read_csv("customer_complaints.csv")

print("Dataset Preview:")
print(df.head())

# ---------------------------
# 2. Encode Categorical Data
# ---------------------------
le_issue = LabelEncoder()
le_priority = LabelEncoder()

df['issue_type_encoded'] = le_issue.fit_transform(df['issue_type'])
df['priority_encoded'] = le_priority.fit_transform(df['priority'])

# Features & Target
X = df[['issue_type_encoded', 'priority_encoded', 'resolution_time_hours']]
y = df['resolved_within_sla']

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Logistic Regression
# ---------------------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ---------------------------
# 5. Random Forest Classifier
# ---------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ---------------------------
# 6. Predict New Complaint
# ---------------------------
new_data = pd.DataFrame({
    'issue_type': ['Technical'],
    'priority': ['High'],
    'resolution_time_hours': [8]
})

# Encode new data
new_data['issue_type_encoded'] = le_issue.transform(new_data['issue_type'])
new_data['priority_encoded'] = le_priority.transform(new_data['priority'])

X_new = new_data[['issue_type_encoded', 'priority_encoded', 'resolution_time_hours']]

# Predictions
log_prediction = log_model.predict(X_new)
rf_prediction = rf_model.predict(X_new)

print("\n=== New Complaint Prediction ===")
print("Logistic Regression:", log_prediction[0])
print("Random Forest:", rf_prediction[0])

# 1 = Resolved within SLA
# 0 = Not resolved within SLA
