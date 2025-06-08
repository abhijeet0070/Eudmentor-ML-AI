import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === Load Data ===
df = pd.read_csv(r"G:\Eductor\data\edu_mentor_dataset_final.csv")

# === Define Categorical Mappings ===
cat_features = ['learning_style', 'content_type_preference', 'teacher_comments_summary']
label_encoders = {}

# Encode categorical columns
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# === Regression Feature Columns ===
REG_FEATURE_COLS = [
    'std', 'math_grade', 'english_grade', 'science_grade', 'history_grade',
    'overall_grade', 'assignment_completion', 'engagement_score',
    'math_lec_present', 'science_lec_present', 'history_lec_present', 'english_lec_present',
    'attendance_ratio', 'login_frequency_per_week', 'average_session_duration_minutes',
    'learning_style', 'content_type_preference', 'completed_lessons',
    'practice_tests_taken', 'lms_test_scores'
]

TARGET_REG = 'risk_score'

# === Prepare Data ===
df = df.dropna(subset=REG_FEATURE_COLS + [TARGET_REG])
X_reg = df[REG_FEATURE_COLS]
y_reg = df[TARGET_REG]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# === Train Regressor ===
reg_model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
reg_model.fit(X_train, y_train)

# === Evaluate ===
y_pred = reg_model.predict(X_test)
print(f"✅ RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")

# === Save Model and Encoders ===
joblib.dump(reg_model, 'ml_model/regression_model.pkl')
joblib.dump(label_encoders, 'ml_model/label_encoders.pkl')
