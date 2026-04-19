# 🫀 Heart Disease ML Predictor

A web-based medical diagnostic tool that uses **Machine Learning** to estimate the probability of heart disease based on clinical indicators. This project demonstrates the application of **Dimensionality Reduction (PCA)** and **Ensemble Learning (Random Forest)** in a client-side environment.

## 🚀 [View Live Demo](https://yugp6488-commits.github.io/Heart_disease_predictor/)

---

## 📊 Overview
This application takes 5 key clinical features and processes them through a pre-trained logic model to predict the risk of heart disease. It features a real-time **PCA Projection Chart** that visualizes where a specific patient falls within the known clusters of healthy vs. diseased patients.

### Key Features:
* **PCA Visualization:** Transforms high-dimensional clinical data into a 2D space for patient clustering analysis.
* **Predictive Modeling:** Uses Random Forest logic (100 Decision Trees) to calculate risk probability.
* **Interactive UI:** Built with Tailwind CSS for a modern, responsive medical dashboard.
* **Serverless:** Runs entirely in the browser via JavaScript.

---

## 🛠️ Technical Stack
* **Frontend:** HTML5, Tailwind CSS, JavaScript (ES6+)
* **Charts:** [Chart.js](https://www.chartjs.org/) for PCA scatter plots.
* **Dataset:** UCI Heart Disease Dataset (sourced via Kaggle).
* **Deployment:** GitHub Pages.

---

## 🧬 Machine Learning Pipeline
1.  **Data Preprocessing:** Standard scaling of clinical features (Age, Cholesterol, Max HR, etc.).
2.  **PCA (Principal Component Analysis):** Reducing the feature space into 2 Principal Components to capture maximum variance.
3.  **Random Forest Classifier:** An ensemble of decision trees used to output a probability score between 0% and 100%.

---

## 📂 Project Structure
```bash
.
├── index.html        # Main application (Logic, UI, and Styles)
└── README.md         # Project documentation
