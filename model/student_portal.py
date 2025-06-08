# student_portal.py
import streamlit as st
import pandas as pd
import joblib

DATA_PATH = r"G:\Eductor\data\edu_mentor_dataset_final.csv"
REG_MODEL_PATH = r"G:\Eductor\ml_model\regression_model.pkl"
CLF_MODEL_PATH = r"G:\Eductor\ml_model\classification.pkl"

@st.cache_resource
def load_models():
    reg_model = joblib.load(REG_MODEL_PATH)
    clf_model = joblib.load(CLF_MODEL_PATH)
    return reg_model, clf_model

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['student_id'] = df['student_id'].astype(str)
    return df

def recommend_resources(grades):
    recommendations = []
    SUBJECT_RESOURCES = {
        "math_grade": [
            ("Khan Academy Math", "https://www.khanacademy.org/math"),
            ("PatrickJMT", "https://www.youtube.com/user/patrickJMT")
        ],
        "english_grade": [
            ("BBC Learning English", "https://www.bbc.co.uk/learningenglish"),
            ("EnglishClass101", "https://www.youtube.com/user/ENGLISHCLASS101")
        ],
        "science_grade": [
            ("CrashCourse Science", "https://www.youtube.com/playlist?list=PL8dPuuaLjXtOfse2ncvffeelTrqvhrzdz"),
            ("SciShow", "https://www.youtube.com/user/scishow")
        ],
        "history_grade": [
            ("CrashCourse History", "https://www.youtube.com/playlist?list=PL8dPuuaLjXtNgK6MZucdYldNkMybYIHKR"),
            ("History Channel", "https://www.youtube.com/user/HistoryChannel")
        ]
    }
    for subject, grade in grades.items():
        if grade < 60:
            recommendations.extend(SUBJECT_RESOURCES.get(subject, []))
    return recommendations

def main():
    st.title("🎓 Student Portal")
    df = load_data()
    reg_model, clf_model = load_models()

    student_id = st.text_input("Enter Student ID:")
    if student_id:
        student_data = df[df["student_id"] == student_id]
        if not student_data.empty:
            st.write("### Student Profile")
            st.dataframe(student_data)
            
            X = student_data.drop(columns=["student_id"])
            predicted_score = reg_model.predict(X)[0]
            dropout_risk = clf_model.predict(X)[0]
            
            st.metric("📊 Predicted Final Score", f"{predicted_score:.2f}")
            st.metric("⚠️ Dropout Risk", "Yes" if dropout_risk else "No")

            st.write("### 📚 Personalized Learning Resources")
            grades = {
                "math_grade": student_data["math_grade"].values[0],
                "english_grade": student_data["english_grade"].values[0],
                "science_grade": student_data["science_grade"].values[0],
                "history_grade": student_data["history_grade"].values[0]
            }
            recommendations = recommend_resources(grades)
            if recommendations:
                for title, link in recommendations:
                    st.markdown(f"- [{title}]({link})")
            else:
                st.write("✅ No additional resources needed!")
        else:
            st.warning("Student ID not found.")
