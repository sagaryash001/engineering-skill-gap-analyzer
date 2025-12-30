import streamlit as st
import joblib
import pandas as pd

# Load trained model and vectorizer
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "output", "job_classifier_model.pkl")
)
vectorizer = joblib.load(
    os.path.join(BASE_DIR, "output", "tfidf_vectorizer.pkl")
)


st.set_page_config(page_title="Engineering Skill Gap Analyzer", layout="centered")

st.title("📊 Engineering Skill Gap Analyzer")
st.write("Predict job category and employability readiness using ML")

# User input
job_text = st.text_area(
    "Paste a Job Description here:",
    height=200
)

if st.button("Analyze Job"):
    if job_text.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Predict job category
        job_vec = vectorizer.transform([job_text])
        category = model.predict(job_vec)[0]

        # Simple employability logic (demo)
        # Skill-based readiness estimation
        skills = [
            "python", "java", "sql", "machine learning", "excel",
            "django", "flask", "aws", "data analysis", "git"
        ]

        job_text_lower = job_text.lower()
        matched_skills = [s for s in skills if s in job_text_lower]
        skill_count = len(matched_skills)

        if skill_count <= 1:
            readiness = "Low"
        elif skill_count <= 3:
            readiness = "Medium"
        else:
            readiness = "High"

        st.write("🛠️ Detected Skills:", matched_skills)


        st.success(f"🔍 Predicted Job Category: **{category}**")
        st.info(f"📈 Estimated Employability Readiness: **{readiness}**")

# Show dataset insights
st.subheader("📌 Industry Insights")

df_skills = pd.read_csv("output/industry_top_skills.csv")
st.dataframe(df_skills.head(10))
st.write("""
This application analyzes job descriptions using a trained machine learning model
to predict job categories and estimate employability readiness.
It is based on real Indian job market data and highlights the skill gap faced by
engineering graduates.
""")
