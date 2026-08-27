import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.write("Enter student information to predict expected final marks.")
st.divider()

DATA_FILE = "train_6UuJx_CVtuZ9l.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def train_model(data):
    features = [
        "StudyHours",
        "Attendance",
        "PreviousMarks",
        "Assignments",
        "InternalMarks",
        "SleepHours",
        "Extracurricular"
    ]

    X = data[features]
    y = data["FinalMarks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2

try:
    data = load_data()
    model, mae, r2 = train_model(data)
except FileNotFoundError:
    st.error(f"Dataset '{DATA_FILE}' was not found. Keep it in the same folder as app.py.")
    st.stop()

st.sidebar.header("📊 Model Information")
st.sidebar.write("Model: Random Forest Regressor")
st.sidebar.write(f"MAE: {mae:.2f}")
st.sidebar.write(f"R² Score: {r2:.2f}")
st.sidebar.info("The model is trained using the included CSV dataset.")

st.header("📝 Student Information")

col1, col2 = st.columns(2)

with col1:
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

    study_hours = st.slider(
        "Daily Study Hours", 0.0, 12.0, 5.0, 0.5
    )

    attendance = st.slider(
        "Attendance (%)", 0, 100, 75
    )

    previous_marks = st.slider(
        "Previous Exam Marks", 0, 100, 60
    )

with col2:
    assignments = st.slider(
        "Assignment Score (%)", 0, 100, 70
    )

    internal_marks = st.slider(
        "Internal Assessment Marks", 0, 100, 65
    )

    sleep_hours = st.slider(
        "Average Sleep Hours", 0.0, 12.0, 7.0, 0.5
    )

    extracurricular = st.slider(
        "Extracurricular Activity (hours/week)", 0.0, 20.0, 5.0, 0.5
    )

st.divider()

if st.button("🔮 Predict Student Performance", type="primary", use_container_width=True):

    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "Attendance": [attendance],
        "PreviousMarks": [previous_marks],
        "Assignments": [assignments],
        "InternalMarks": [internal_marks],
        "SleepHours": [sleep_hours],
        "Extracurricular": [extracurricular]
    })

    predicted_marks = float(model.predict(input_data)[0])
    predicted_marks = float(np.clip(predicted_marks, 0, 100))

    if predicted_marks >= 85:
        performance = "Excellent 🌟"
        status = "Pass"
    elif predicted_marks >= 70:
        performance = "Very Good 👍"
        status = "Pass"
    elif predicted_marks >= 50:
        performance = "Good 🙂"
        status = "Pass"
    elif predicted_marks >= 40:
        performance = "Average ⚠️"
        status = "Pass"
    else:
        performance = "Needs Improvement ❌"
        status = "Fail"

    st.success("Prediction completed successfully!")

    st.header("📈 Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Predicted Final Marks", f"{predicted_marks:.1f}/100")

    with c2:
        st.metric("Result", status)

    with c3:
        st.metric("Performance", performance)

    st.subheader("Performance Score")
    st.progress(int(predicted_marks))

    st.subheader("💡 Recommendations")
    recommendations = []

    if study_hours < 4:
        recommendations.append("Increase daily study time to at least 4–5 hours.")

    if attendance < 75:
        recommendations.append("Improve attendance and try to maintain at least 75%.")

    if previous_marks < 50:
        recommendations.append("Focus on fundamentals and previously weak topics.")

    if assignments < 60:
        recommendations.append("Complete assignments regularly and improve assignment scores.")

    if internal_marks < 50:
        recommendations.append("Prepare better for internal assessments.")

    if sleep_hours < 6:
        recommendations.append("Maintain a healthy sleep schedule of around 7–8 hours.")

    if not recommendations:
        recommendations.append("Excellent! Maintain your current study habits and consistency.")

    for item in recommendations:
        st.write("• " + item)

    st.subheader("📋 Student Data")

    result_table = pd.DataFrame({
        "Parameter": [
            "Student Name",
            "Study Hours",
            "Attendance",
            "Previous Marks",
            "Assignment Score",
            "Internal Marks",
            "Sleep Hours",
            "Extracurricular Hours"
        ],
        "Value": [
            student_name or "Not Provided",
            f"{study_hours} hours/day",
            f"{attendance}%",
            f"{previous_marks}/100",
            f"{assignments}%",
            f"{internal_marks}/100",
            f"{sleep_hours} hours",
            f"{extracurricular} hours/week"
        ]
    })

    st.table(result_table)

st.divider()
st.caption("Student Performance Prediction System | Machine Learning + Streamlit")
