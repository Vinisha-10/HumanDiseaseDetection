# 🧠 Human Disease Detection

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Streamlit-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-5%20Models-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Live-success?style=for-the-badge">
</p>

<p align="center">
  ⚡ Predict diseases instantly based on symptoms using Machine Learning  
</p>

---

## ✨ Overview

**Human Disease Detection** is a smart web application that predicts potential diseases based on user-provided symptoms.

Built with **Streamlit**, it leverages multiple machine learning models to deliver fast and reliable predictions — all through an intuitive interface.

---

## 🚀 Live Demo

🔗 **Try it here:**  
👉 https://humandiseasedetectionbyvinisha.streamlit.app/

⚠️ **Note:** The app may take ~30 seconds to load on first open (it goes to sleep when inactive).

---

## 🎯 Key Features

✨ **Interactive UI**  
> Simple and user-friendly interface powered by Streamlit  

🤖 **Multiple ML Models**  
> Uses:
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- Naive Bayes
- K-Nearest Neighbors (KNN)

⚡ **Real-Time Predictions**  
> Instant disease prediction based on selected symptoms  

📊 **Visual Insights**  
> Comparison graphs showing agreement across models  

---

## 🧩 Project Structure

```bash
Human-Disease-Detection/
│
├── streamlit_app.py        # Main app interface
├── models_training.ipynb   # Model training & evaluation
│
├── datasets/               # Training data
│   └── symptoms_disease.csv
│
├── artifacts/              # Saved ML models
│   └── *.pkl
│
└── README.md

## ⚙️ How It Works

```mermaid
flowchart LR
A[User Inputs Symptoms] --> B[Streamlit Interface]
B --> C[ML Models]
C --> D[Predictions Generated]
D --> E[Comparison Graphs]