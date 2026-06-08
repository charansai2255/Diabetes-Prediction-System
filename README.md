# 🩺 Diabetes Prediction System

A Machine Learning-based web application that predicts whether a person is likely to have diabetes based on health-related input parameters.

The application is built using **Python**, **Scikit-Learn**, and **Streamlit**, and uses a trained **Support Vector Machine (SVM)** model for prediction.

## 🚀 Features

* Diabetes prediction using Machine Learning
* Support Vector Machine (SVM) model
* Data preprocessing with StandardScaler
* Interactive web interface
* Real-time prediction results
* Easy deployment with Streamlit

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-Learn
* NumPy
* Pandas
* Joblib

## 📂 Project Structure

```text
Diabetes-Prediction-System/
│
├── app.py
├── diabetes_svm_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Diabetes-Prediction-System.git
cd Diabetes-Prediction-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Open in Browser

```text
http://localhost:8501
```

## 📊 Input Features

The model uses the following health parameters:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

## 🎯 Prediction Output

The application predicts:

* Diabetic
* Not Diabetic

and displays a risk score based on the trained model.

## ⚠️ Disclaimer

This application is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 👨‍💻 Author

Your Name

GitHub: https://github.com/charansai2255
