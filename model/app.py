# app.py
import streamlit as st
from student_portal import main as student_main
from teacher_portal import main as teacher_main

st.set_page_config(page_title="EduMentor - Teacher Panel", layout="wide")

if "view" not in st.session_state:
    st.session_state["view"] = "Student"

st.sidebar.title("📘 EduMentor App")
selected_view = st.sidebar.radio("Select View:", ["Student", "Teacher"])

if selected_view != st.session_state["view"]:
    st.session_state["view"] = selected_view
    st.rerun()

if st.session_state["view"] == "Student":
    student_main()
else:
    teacher_main()



