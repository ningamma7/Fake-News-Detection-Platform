# 📰 Fake News Detection Platform
### Presidency University | CSE7102 - Mini Project | Review-1

> Offline, Zero-Cost Fake News Detection Platform using TF-IDF + DistilBERT trained on 44,898 news articles with Confidence Scoring & Suspicious Word Highlighting.

---

### 🎯 Problem Statement
The spread of fake news on social media misleads public opinion and creates confusion during critical events. There is a need for an offline, low-cost system that can automatically detect fake news and explain why it is fake.

### 🎯 Objectives for Review-1
1. Data Collection & Pre-processing (Fake.csv + True.csv)
2. TF-IDF Feature Extraction + Baseline Model (Logistic Regression)
3. Basic Flask Web Platform with Input & Output
4. GitHub Team Collaboration Setup

### ✨ Features (Review-1 Completed)
- [x] Dataset: Fake.csv (23,481) + True.csv (21,417) = 44,898 articles
- [x] TF-IDF Vectorizer (max_features=5000)
- [x] Logistic Regression Baseline - 89% Accuracy
- [x] Flask App with Fake/Real Prediction
- [x] Confidence Score (%) + Uncertain Label
- [x] Suspicious Word Highlighting (Red)
- [x] SQLite History Tracking
- [ ] DistilBERT Fine-tuning (For Review-2)
- [ ] Final Deployment (For Review-2)

### 🛠️ Tech Stack - Zero Cost, 100% Offline
- ML:Python, Scikit-learn, TF-IDF, Logistic Regression, DistilBERT
- Backend: Flask, SQLite3
- Frontend: HTML, CSS, Jinja2
- Dataset: Kaggle Fake and True News Dataset
- Cost: $0 - No API, No Cloud

### 👥 Team Members - CSE - Presidency University

| Name | USN | Role | GitHub Profile | Contribution |
| :--- | :--- | :--- | :--- | :--- |
| [Pragnya Dattatri] | 20231IST0080 | Model Training & BERT Integration | [@pragnyadattatri-create](https://github.com/pragnyadattatri-create) | train.py, model/ |
| Harshitha P | 20231IST0082 | Frontend UI & Highlighting Logic | [@Harshitha1024](https://github.com/Harshitha1024) | templates/, static/ |
|  Ningamma Mariyajjanavara | 20231IST0087 | Flask Backend & SQLite DB | [@ningammam7](https://github.com/ningammam7) | app.py, database.db, README, Testing |

Guide: Prof. [Mr. Paruchuru Jyothi Satya Naga Swaroop]
Course Code: CSE7102
Review Date:29th August 2026

### 📁 Project Structure (Review-1)
