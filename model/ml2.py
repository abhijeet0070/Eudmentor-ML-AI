import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import randint, uniform

# --- 1) Load raw data ---
df = pd.read_csv(r"G:\Eductor\data\edu_mentor_dataset_final.csv")

# --- 2) Encode categorical columns ---
cat_features = ['learning_style', 'content_type_preference', 'teacher_comments_summary']
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- 3) Define feature sets and targets ---
ALL_FEATURES = [
    'std',
    'math_grade', 'english_grade', 'science_grade', 'history_grade',
    'overall_grade', 'assignment_completion', 'engagement_score',
    'math_lec_present', 'science_lec_present', 'history_lec_present', 'english_lec_present',
    'attendance_ratio', 'login_frequency_per_week', 'average_session_duration_minutes',
    'learning_style', 'content_type_preference', 'completed_lessons',
    'practice_tests_taken', 'lms_test_scores', 'teacher_comments_summary'
]

# Regression uses all 21 features
X_reg = df[ALL_FEATURES]
y_reg = df['risk_score']

# Classification uses the same 21 features
X_clf = df[ALL_FEATURES]
y_clf = df['is_at_risk']

# Drop any rows with missing values in features or targets
df_clean = df.dropna(subset=ALL_FEATURES + ['risk_score', 'is_at_risk'])
X_reg = df_clean[ALL_FEATURES]
y_reg = df_clean['risk_score']
X_clf = df_clean[ALL_FEATURES]
y_clf = df_clean['is_at_risk']

# --- 4) Train/test split ---
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# --- 5a) Train XGB Regression with simple hyperparam tuning ---
reg = XGBRegressor(random_state=42, objective='reg:squarederror')
param_dist_reg = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}
rs_reg = RandomizedSearchCV(reg, param_dist_reg, n_iter=20, cv=3, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
rs_reg.fit(Xr_train, yr_train)
best_reg = rs_reg.best_estimator_

# Evaluate regression
yr_pred = best_reg.predict(Xr_test)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
print(f"[Regression] RMSE: {rmse:.2f}, R²: {r2:.3f}")

# --- 5b) Train XGB Classification ---
clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
param_dist_clf = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}
rs_clf = RandomizedSearchCV(clf, param_dist_clf, n_iter=20, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
rs_clf.fit(Xc_train, yc_train)
best_clf = rs_clf.best_estimator_

# Evaluate classification
yc_pred = best_clf.predict(Xc_test)
acc = accuracy_score(yc_test, yc_pred)
print(f"[Classification] Accuracy: {acc:.3f}")
print("Classification report:\n", classification_report(yc_test, yc_pred))
print("Confusion matrix:\n", confusion_matrix(yc_test, yc_pred))

# --- 6) Save models and encoders ---
# joblib.dump(best_reg,  r"G:\Eductor\ml_model\regression_model.pkl")
# joblib.dump(best_clf, r"G:\Eductor\ml_model\classification.pkl")
# joblib.dump(label_encoders, r"G:\Eductor\ml_model\label_encoders.pkl")

# print("✅ All models and encoders saved.")
