import streamlit as st
import pandas as pd
import pickle
import os

# ---------- CONFIG ----------
DATA_PATH = r"G:\Eductor\data\edu_mentor_dataset_final.csv"
NEW_DATA_PATH = r"G:\Eductor\data\new_students.csv"
REGRESSION_MODEL_PATH = r"G:\Eductor\ml_model\regression_model.pkl"
CLASSIFICATION_MODEL_PATH = r"G:\Eductor\ml_model\classification.pkl"

FEATURE_COLS = [
    'std', 'math_grade', 'english_grade', 'science_grade', 'history_grade',
    'overall_grade', 'assignment_completion', 'engagement_score',
    'math_lec_present', 'science_lec_present', 'history_lec_present', 'english_lec_present',
    'attendance_ratio', 'login_frequency_per_week', 'average_session_duration_minutes',
    'learning_style', 'content_type_preference', 'completed_lessons',
    'practice_tests_taken', 'lms_test_scores', 'teacher_comments_summary'
]

TEACHER_USERNAME = "teacher"
TEACHER_PASSWORD = "abhi123"

# Manual encodings matching training
learning_style_map = {"Visual": 0, "Auditory": 1, "Kinesthetic": 2}
content_type_map = {"Video": 0, "Text": 1, "Interactive": 2}
teacher_comments_map = {"Needs Improvement": 0, "Average": 1, "Good": 2, "Excellent": 3}

# -------- CACHING --------
@st.cache_resource
def load_models():
    with open(REGRESSION_MODEL_PATH, 'rb') as f:
        reg_model = pickle.load(f)
    with open(CLASSIFICATION_MODEL_PATH, 'rb') as f:
        clf_model = pickle.load(f)
    return reg_model, clf_model

@st.cache_data
def load_data():
    # Load main + new students if new students file exists
    df_main = pd.read_csv(DATA_PATH, low_memory=False)
    df_main['student_id'] = df_main['student_id'].astype(str)
    
    if os.path.exists(NEW_DATA_PATH):
        df_new = pd.read_csv(NEW_DATA_PATH, low_memory=False)
        df_new['student_id'] = df_new['student_id'].astype(str)
        df = pd.concat([df_main, df_new], ignore_index=True)
    else:
        df = df_main
    return df

def save_student(new_student: pd.DataFrame):
    if os.path.exists(NEW_DATA_PATH):
        df = pd.read_csv(NEW_DATA_PATH, low_memory=False)
        df = pd.concat([df, new_student], ignore_index=True)
    else:
        df = new_student
    df.to_csv(NEW_DATA_PATH, index=False)

# -------- UI COMPONENTS --------

def login():
    st.sidebar.title("👩‍🏫 Teacher Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")
    if login_button:
        if username == TEACHER_USERNAME and password == TEACHER_PASSWORD:
            st.session_state['logged_in'] = True
            st.sidebar.success("✅ Login successful!")
        else:
            st.sidebar.error("❌ Invalid credentials")

def add_student_form(reg_model, clf_model):
    st.subheader("➕ Add New Student")

    with st.form("add_student_form"):
        student_id = st.text_input("Student ID")
        name = st.text_input("Name")
        std = st.number_input("Standard/Class", 1, 12, step=1)
        math_grade = st.slider("Math Grade", 0, 100)
        english_grade = st.slider("English Grade", 0, 100)
        science_grade = st.slider("Science Grade", 0, 100)
        history_grade = st.slider("History Grade", 0, 100)
        overall_grade = st.slider("Overall Grade", 0, 100)
        assignment_completion = st.slider("Assignment Completion (%)", 0, 100)
        engagement_score = st.slider("Engagement Score", 0, 10)
        math_lec_present = st.slider("Math Lectures Attended", 0, 50)
        science_lec_present = st.slider("Science Lectures Attended", 0, 50)
        history_lec_present = st.slider("History Lectures Attended", 0, 50)
        english_lec_present = st.slider("English Lectures Attended", 0, 50)
        attendance_ratio = st.slider("Attendance Ratio", 0.0, 1.0, 0.01)
        login_frequency_per_week = st.slider("Logins Per Week", 0, 14)
        average_session_duration_minutes = st.slider("Avg Session Duration (min)", 0, 300)
        completed_lessons = st.number_input("Completed Lessons", 0)
        practice_tests_taken = st.number_input("Practice Tests Taken", 0)
        lms_test_scores = st.slider("LMS Test Score (%)", 0, 100)

        learning_style = st.selectbox("Learning Style", list(learning_style_map.keys()))
        content_type_preference = st.selectbox("Content Type Preference", list(content_type_map.keys()))
        teacher_comments_summary = st.selectbox("Teacher Comment Summary", list(teacher_comments_map.keys()))

        submit = st.form_submit_button("Add Student")

    if submit:
        input_data = pd.DataFrame([[
            std, math_grade, english_grade, science_grade, history_grade,
            overall_grade, assignment_completion, engagement_score,
            math_lec_present, science_lec_present, history_lec_present, english_lec_present,
            attendance_ratio, login_frequency_per_week, average_session_duration_minutes,
            learning_style_map[learning_style],
            content_type_map[content_type_preference],
            completed_lessons, practice_tests_taken, lms_test_scores,
            teacher_comments_map[teacher_comments_summary]
        ]], columns=FEATURE_COLS)

        try:
            risk_score = reg_model.predict(input_data)[0]
            is_at_risk = clf_model.predict(input_data)[0]

            new_student = pd.DataFrame([{
                "student_id": student_id,
                "name": name,
                **dict(zip(FEATURE_COLS, input_data.iloc[0])),
                "risk_score": round(risk_score, 2),
                "is_at_risk": int(is_at_risk)
            }])

            save_student(new_student)
            st.success(f"Student {name} added successfully with risk_score: {risk_score:.2f} and risk classification: {'At Risk' if is_at_risk else 'Not At Risk'}")

        except Exception as e:
            st.error(f"Error during prediction or saving: {e}")

def search_students(df):
    st.subheader("🔍 Search Students")

    student_id = st.text_input("Enter Student ID to search")
    if st.button("Search"):
        if student_id:
            results = df[df['student_id'] == student_id]
            if not results.empty:
                st.write("### Search Results:")
                st.dataframe(results)
            else:
                st.warning("No student found with this ID.")
        else:
            st.warning("Please enter a Student ID.")

def delete_student():
    st.subheader("🗑 Delete Student")
    student_id = st.text_input("Enter Student ID to delete")
    if st.button("Delete"):
        if student_id:
            if os.path.exists(NEW_DATA_PATH):
                df = pd.read_csv(NEW_DATA_PATH, low_memory=False)
                initial_len = len(df)
                df = df[df['student_id'] != student_id]
                if len(df) < initial_len:
                    df.to_csv(NEW_DATA_PATH, index=False)
                    st.success(f"Student with ID {student_id} deleted successfully from new students data.")
                else:
                    st.warning("Student ID not found in new students data.")
            else:
                st.warning("No new students data file found to delete from.")
        else:
            st.warning("Please enter a Student ID.")

def view_students(df):
    st.subheader("📋 View All Students")
    st.dataframe(df)

    # Add a bar chart showing number of At Risk vs Not At Risk
    if 'is_at_risk' in df.columns:
        risk_counts = df['is_at_risk'].value_counts().rename({0: "Not At Risk", 1: "At Risk"})
        st.bar_chart(risk_counts)
    else:
        st.info("Risk classification data not available.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
        return

    reg_model, clf_model = load_models()
    df = load_data()

    st.title("EduMentor Teacher Portal")

    menu = ["Add Student", "Search Students", "Delete Student", "View All Students", "Logout"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Add Student":
        add_student_form(reg_model, clf_model)
    elif choice == "Search Students":
        search_students(df)
    elif choice == "Delete Student":
        delete_student()
    elif choice == "View All Students":
        view_students(df)
    elif choice == "Logout":
        st.session_state['logged_in'] = False
        st.experimental_rerun()