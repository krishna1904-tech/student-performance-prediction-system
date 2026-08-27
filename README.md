# 🎓 Student Performance Prediction System

A machine-learning based Student Performance Prediction System built with **Python, Scikit-learn, Pandas, and Streamlit**.

## Features

- Student information input
- Predicts expected final marks
- Predicts Pass/Fail status
- Shows performance category
- Provides personalized recommendations
- Displays model performance metrics
- Uses a CSV dataset for training

## Project Structure

```text
Student_Performance_Prediction/
├── app.py
├── requirements.txt
├── train_6UuJx_CVtuZ9l.csv
└── README.md
```

## Installation

Open the project folder in VS Code and run:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Dataset

The CSV file contains student academic and lifestyle features such as:

- Study Hours
- Attendance
- Previous Marks
- Assignment Score
- Internal Marks
- Sleep Hours
- Extracurricular Hours
- Final Marks

## Machine Learning Model

The application uses a **Random Forest Regressor** to predict final marks.

> Note: The included dataset is generated sample data for an academic/demo project. For real-world prediction, replace it with a properly collected and ethically handled student dataset.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
