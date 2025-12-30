📊 Engineering Skill Gap Analyzer
📌 Overview

The Engineering Skill Gap Analyzer is a data science and machine learning project that analyzes real Indian job market data to identify the skill and employability gap among engineering graduates.
It uses Natural Language Processing (NLP) and Machine Learning to extract industry skill demand, classify job roles, and compute an employability readiness score.
A Streamlit-based GUI allows interactive analysis and real-time predictions.

🎯 Objectives

Analyze industry skill demand using real job postings

Identify mismatch between academic output and market needs

Predict job category from job descriptions using ML

Quantify employability readiness using a scoring model

Provide an interactive GUI for demonstration

📂 Dataset

Source: Indian job postings dataset (Naukri.com)

Size: ~22,000 job records

Key Features:

Job Title

Job Description

Required Skills

Experience Level

This dataset represents real-world industry demand, making the analysis reliable and publishable.

🛠️ Technologies & Tools Used
Category	Tools
Programming	Python
Data Analysis	Pandas, NumPy
Visualization	Matplotlib
NLP	TF-IDF Vectorization
Machine Learning	Scikit-learn (Logistic Regression)
Model Storage	Joblib
GUI	Streamlit
Version Control	Git, GitHub
🧠 Methodology

Data Cleaning & Preprocessing

Removed missing values

Cleaned skill text using regex

Exploratory Data Analysis

Identified top in-demand skills

Analyzed experience requirements

Skill Demand Analysis

Frequency-based extraction of industry skills

Visualization of top skills

Machine Learning Model

TF-IDF vectorization of job descriptions

Logistic Regression classifier

Train / validation / test split

Hyperparameter tuning using GridSearchCV

Employability Score Model

Composite score based on:

Experience level

Skill count

Categorized into Low / Medium / High readiness

GUI Development

Streamlit-based interactive interface

Real-time prediction and insights

🤖 Machine Learning Details

Model: Logistic Regression

Features: TF-IDF vectors from job descriptions

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

The trained model and vectorizer are saved using Joblib for reuse and deployment.

🖥️ GUI (Streamlit App)

The GUI allows users to:

Paste a job description

Predict job category (IT / HR / Sales / Finance)

View employability readiness estimate

Explore top industry skills
