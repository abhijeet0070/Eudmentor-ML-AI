import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
from io import BytesIO

# --- Constants ---
DATA_PATH = r"G:\Eductor\data\edu_mentor_dataset_final.csv"
REG_MODEL_PATH = r"G:\Eductor\ml_model\regression_model.pkl"
CLF_MODEL_PATH = r"G:\Eductor\ml_model\classification.pkl"
ENCODER_PATH = r"G:\Eductor\ml_model\label_encoders.pkl"

FEATURE_COLS = [
    'std', 'math_grade', 'english_grade', 'science_grade', 'history_grade',
    'overall_grade', 'assignment_completion', 'engagement_score',
    'math_lec_present', 'science_lec_present', 'history_lec_present', 'english_lec_present',
    'attendance_ratio', 'login_frequency_per_week', 'average_session_duration_minutes',
    'learning_style', 'content_type_preference', 'completed_lessons',
    'practice_tests_taken', 'lms_test_scores', 'teacher_comments_summary'
]

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
    ],
    "computer_science_grade": [
        ("freeCodeCamp Computer Science", "https://www.freecodecamp.org/learn/computer-science/"),
        ("CS50 by Harvard", "https://cs50.harvard.edu/x/2023/")
    ],
    "geography_grade": [
        ("National Geographic Kids", "https://kids.nationalgeographic.com/"),
        ("Geography Now", "https://www.youtube.com/user/GeographyNow")
    ],
    "physics_grade": [
        ("MinutePhysics", "https://www.youtube.com/user/minutephysics"),
        ("Physics Girl", "https://www.youtube.com/user/physicswoman")
    ],
    "chemistry_grade": [
        ("Khan Academy Chemistry", "https://www.khanacademy.org/science/chemistry"),
        ("Periodic Videos", "https://www.youtube.com/user/periodicvideos")
    ],
    "biology_grade": [
        ("Bozeman Science", "https://www.bozemanscience.com/biology"),
        ("Amoeba Sisters", "https://www.youtube.com/user/AmoebaSisters")
    ],
    "economics_grade": [
        ("Economics Explained", "https://www.youtube.com/c/EconomicsExplained"),
        ("Khan Academy Economics", "https://www.khanacademy.org/economics-finance-domain")
    ]
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['student_id'] = df['student_id'].astype(str)
    return df

@st.cache_resource
def load_models():
    return (
        joblib.load(REG_MODEL_PATH),
        joblib.load(CLF_MODEL_PATH),
        joblib.load(ENCODER_PATH)
    )

def generate_pdf(student_name, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Student Report for {student_name}", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, report_text)
    pdf_output = BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    return pdf_output

def main():
    icon_path = r"G:\Eductor\download.png"
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(icon_path, width=50)
    with col2:
        st.title("EduMentor - Student Risk Predictor")

    df = load_data()
    reg_model, clf_model, encoders = load_models()

    student_id = st.text_input("Enter Student ID")
    student_name = st.text_input("Enter Student Name")

    if student_id and student_name:
        student = df[(df['student_id'] == student_id) & (df['student_name'].str.lower() == student_name.lower())]

        if student.empty:
            st.error("Student not found")
            return

        st.success("Student Found")
        st.dataframe(student[["student_id", "student_name", "risk_score", "is_at_risk"]])

        features = student[FEATURE_COLS].copy()
        for col in ['learning_style', 'content_type_preference', 'teacher_comments_summary']:
            if col in encoders:
                features[col] = encoders[col].transform(features[col].astype(str))

        X = np.array(features, dtype=float)
        risk_score_pred = reg_model.predict(X)[0]
        risk_status_pred = clf_model.predict(X)[0]
        risk_prob = clf_model.predict_proba(X)[0][1]

        st.subheader("Prediction Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Risk Score", f"{risk_score_pred:.2f}")
        c2.metric("Predicted Risk", "At Risk" if risk_status_pred == 1 else "Not At Risk")
        c3.metric("Probability", f"{risk_prob:.2f}")

        attendance = student["attendance_ratio"].values[0]
        st.write(f"*Attendance Ratio:* {attendance:.2f}")
        if attendance < 0.75:
            st.warning("Low Attendance Warning!")

        st.subheader("Improvement Suggestions")
        report_text = (
            f"Student Name: {student_name}\n"
            f"Student ID: {student_id}\n"
            f"Predicted Risk Score: {risk_score_pred:.2f}\n"
            f"Risk Status: {'At Risk' if risk_status_pred == 1 else 'Not At Risk'}\n"
            f"Probability: {risk_prob:.2f}\n\nSuggestions:\n"
        )

        grade_cols = [col for col in student.columns if col.endswith('_grade')]
        for subj in grade_cols:
            try:
                score = float(student[subj].values[0])
            except:
                continue

            subj_name = subj.replace('grade', '').replace('', ' ').title()
            st.write(f"{subj_name} Score:** {score}")
            if score < 70:
                st.info(f"Needs Improvement in {subj_name}")
                report_text += f"- {subj_name} below 70 (Score: {score}).\n"
                if subj in SUBJECT_RESOURCES:
                    st.markdown("Recommended Resources:")
                    for name, url in SUBJECT_RESOURCES[subj]:
                        st.markdown(f"- [{name}]({url})")
                        report_text += f"  - {name}: {url}\n"
                else:
                    st.warning("No resource links available for this subject.")

        if st.button("📄 Download PDF Report"):
            pdf_file = generate_pdf(student_name, report_text)
            st.download_button(
                label="Download Report PDF",
                data=pdf_file,
                file_name=f"{student_name}_report.pdf",
                mime="application/pdf"
            )

if _name_ == "_main_":
    main()
