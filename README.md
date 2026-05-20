# 🚨 Clinical Sepsis Early-Warning Dashboard

[![Live App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://sepsis-risk-ai.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/XYQ493/Sepsis-Early-Warning-Dashboard)

An enterprise-grade predictive system utilizing an optimized Random Forest classifier to assess patient systemic infection risk based on core physiological and laboratory biomarkers.

🔗 **Live Deployment Link:** [https://sepsis-risk-ai.streamlit.app](https://sepsis-risk-ai.streamlit.app)

---

## 📊 Project Overview
Sepsis remains one of the leading causes of mortality in Intensive Care Units (ICUs) globally. Early clinical identification is paramount, as every hour of delayed treatment significantly increases mortality risks. 

This project solves this critical clinical challenge by translating complex, non-linear machine learning predictive analytics into an intuitive, real-time interactive dashboard designed for clinical decision support. Healthcare professionals can input real-time physiological metrics to instantly compute patient sepsis risk probabilities.

## 🛠️ The Tech Stack
* **Language:** Python 3.14
* **Interactive Dashboard:** Streamlit Community Cloud
* **Machine Learning Pipelines:** Scikit-Learn (Random Forest Architecture), XGBoost
* **Data Processing & Analytics:** Pandas, NumPy
* **Model Serialization:** Joblib / Pickle

## 📈 Model Performance & Architecture
The underlying core engine utilizes an optimized **Random Forest Classifier** trained on rigorous ICU patient cohorts. 
* **Evaluation Metrics:** Achieved high sensitivity (Recall) and robust ROC-AUC scores, ensuring the minimization of false negatives—a critical requirement in medical diagnostics.
* **Non-Linear Dynamics:** The pipeline successfully models intricate interactions between key clinical variables (e.g., Lactate thresholds, SOFA scores, and vital stat changes) to output fine-grained probability percentages rather than generic binary flags.

## 🗂️ Repository Structure
* `app.py` - The core frontend interface and model execution script.
* `models/` - Production-ready serialized machine learning model files (`.pkl`/`.joblib`).
* `requirements.txt` - Python deployment dependencies optimized for Linux-based cloud runtimes.
