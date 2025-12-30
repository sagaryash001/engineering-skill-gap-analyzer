import pandas as pd

df = pd.read_csv("datasets/naukri_jobs.csv")
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

df.head()

print(df.shape)
print(df.columns)

import re

# Keep only useful columns
df = df[['jobtitle', 'experience', 'jobdescription', 'skills']]
df.dropna(inplace=True)

# Clean skills text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z, ]', '', text)
    return text

df['clean_skills'] = df['skills'].apply(clean_text)

print("\nCleaned skills sample:")
print(df['clean_skills'].head())

from collections import Counter
import matplotlib.pyplot as plt

# Split skills by comma
skill_list = []

for skills in df['clean_skills']:
    skill_list.extend([s.strip() for s in skills.split(',') if s.strip() != ''])

# Count skill frequency
skill_counts = Counter(skill_list)

# Convert to DataFrame
skill_df = pd.DataFrame(skill_counts.items(), columns=['Skill', 'Count'])
skill_df = skill_df.sort_values(by='Count', ascending=False)

print("\nTop 10 In-Demand Skills:")
print(skill_df.head(10))
# Clean experience column and create experience_level
def clean_experience(exp):
    exp = str(exp).lower()
    if 'fresher' in exp or '0' in exp:
        return 'Fresher'
    elif '1' in exp or '2' in exp:
        return '0-2 Years'
    elif '3' in exp or '4' in exp or '5' in exp:
        return '3-5 Years'
    else:
        return '5+ Years'

df['experience_level'] = df['experience'].apply(clean_experience)

# Plot top 10 skills
# Count experience levels
exp_counts = df['experience_level'].value_counts()

print("\nExperience level distribution:")
print(exp_counts)
plt.figure(figsize=(6,4))


plt.title("Experience Demand in Indian Job Market")
plt.xlabel("Experience Level")
plt.ylabel("Number of Job Postings")
plt.tight_layout()

plt.savefig("output/experience_demand.png")
print("Chart saved as output/experience_demand.png")

# Save top industry skills to CSV (for report & future comparison)
import os

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

top_skills = skill_df.head(15)
top_skills.to_csv("output/industry_top_skills.csv", index=False)

print("File saved successfully at output/industry_top_skills.csv")

# =====================================================
# ADVANCED EXTENSION: ML + EMPLOYABILITY MODEL
# =====================================================
# -----------------------------
# JOB CATEGORY CREATION
# -----------------------------

def assign_category(text):
    text = str(text).lower()
    if 'software' in text or 'developer' in text or 'programming' in text:
        return 'IT'
    elif 'sales' in text or 'marketing' in text:
        return 'Sales/Marketing'
    elif 'hr' in text or 'human resource' in text:
        return 'HR'
    elif 'account' in text or 'finance' in text:
        return 'Finance'
    else:
        return 'Other'

df['job_category'] = df['jobdescription'].apply(assign_category)

print("\nJob category distribution:")
print(df['job_category'].value_counts())

# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df['jobdescription']
y = df['job_category']

tfidf = TfidfVectorizer(stop_words='english', max_features=300)
X_tfidf = tfidf.fit_transform(X)

# Train (70%), Validation (15%), Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_tfidf, y, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# -----------------------------
# HYPERPARAMETER TUNING
# -----------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest hyperparameters:", grid.best_params_)
# -----------------------------
# MODEL EVALUATION
# -----------------------------
from sklearn.metrics import accuracy_score, classification_report

# Validation performance
val_pred = best_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_pred))

# Test performance
test_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("\nClassification Report:\n",
      classification_report(y_test, test_pred))

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
import joblib

joblib.dump(best_model, "output/job_classifier_model.pkl")
joblib.dump(tfidf, "output/tfidf_vectorizer.pkl")

print("Trained model and vectorizer saved to output/")

# -----------------------------
# EMPLOYABILITY SCORE MODEL
# -----------------------------

def employability_score(exp_level, skill_count):
    score = 0

    if exp_level == 'Fresher':
        score += 1
    elif exp_level == '0-2 Years':
        score += 2
    elif exp_level == '3-5 Years':
        score += 3
    else:
        score += 4

    if skill_count >= 3:
        score += 2
    elif skill_count == 2:
        score += 1

    return score

df['skill_count'] = df['clean_skills'].apply(lambda x: len(x.split(',')))

df['employability_score'] = df.apply(
    lambda row: employability_score(row['experience_level'], row['skill_count']),
    axis=1
)

def readiness_level(score):
    if score <= 2:
        return 'Low'
    elif score <= 4:
        return 'Medium'
    else:
        return 'High'

df['readiness_level'] = df['employability_score'].apply(readiness_level)

df[['jobtitle', 'experience_level', 'skill_count',
    'employability_score', 'readiness_level']] \
    .to_csv("output/employability_scores.csv", index=False)

print("Employability scores saved to output/employability_scores.csv")


