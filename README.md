# Spam Email Classification Project

## Overview

This project is a **Spam Email Classifier** built using multiple machine learning models to detect whether a message is spam or ham (not spam). The application is built with **Streamlit** to provide an interactive web interface for users to:

- Explore the project overview
- Compare model performances
- Test new messages for spam detection

## Features

- Uses five different machine learning models:
  - Logistic Regression
  - Multinomial Naive Bayes (best performing model)
  - Support Vector Machines (SVM)
  - Random Forest
  - XGBoost
- Displays training and testing accuracies for all models
- Visualizes comparison through scatter plots
- Allows users to test new messages with the Naive Bayes model
- Sidebar with navigation and custom image/logo

## Requirements

- Python 3.8+
- Libraries:
  - streamlit
  - scikit-learn
  - xgboost
  - matplotlib
  - joblib
  - pandas
  - pillow (for image display)

You can install all requirements via:

```bash
pip install streamlit scikit-learn xgboost matplotlib joblib pandas pillow



# 📧 Smart Spam Detector

A simple yet powerful Streamlit app for detecting spam messages using machine learning models.

---

## 🧠 Project Overview

This app performs **spam classification** using multiple ML models. It allows users to test new messages and compare model performance, all in a user-friendly interface.

---

## 📊 Model Comparison

- Compares the **train/test accuracy** of different classifiers.
- Displays the results in both tabular and scatter plot formats.
- Highlights the best-performing model.

---

## 📬 Test a Message

- Enter a message in the text area.
- The app uses the **Naive Bayes** model to classify it as **Spam** or **Ham**.

---

## ✅ How It Works

1. The dataset is preprocessed (including **lemmatization**).
2. Text is vectorized using **CountVectorizer**.
3. Multiple models are trained with **tuned hyperparameters**.
4. The best model (**Multinomial Naive Bayes**) is saved and used in the app.
5. User inputs a message → prediction displayed instantly.

---

## 🗂️ Project Structure

```
spam_email_classification/
│
├── spam_email_detectApp.py      # Main Streamlit app
├── spam_model/
│   ├── best_model_mnb.pkl       # Trained Naive Bayes model
│   └── vectorizer.pkl     # CountVectorizer instance
├── images/
│   └── sidebar_logo.png         # Image for sidebar
├── README.md
└── requirements.txt             # Optional: List of dependencies
```

---

## ⚙️ Usage

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/smart-spam-detector.git
cd smart-spam-detector
```

2. **Create & activate a virtual environment (recommended):**

```bash
# Create
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

> Or manually:
```bash
pip install streamlit scikit-learn matplotlib joblib
```

4. **Run the app:**

```bash
streamlit run spam_email_detectApp.py
```

5. **Navigate using the sidebar** to switch between:
   - 🧠 Project Overview
   - 📊 Model Comparison
   - 📬 Test a Message

---

## 📝 Notes

- ✅ Use `use_container_width=True` instead of the deprecated `use_column_width` for images in Streamlit.
- 📁 Make sure all **paths to models and images** are correct.
- ⚠️ Always install dependencies in the **active virtual environment**.

---

## 🌟 Future Improvements

- Add more advanced **text preprocessing** and **feature engineering**.
- Enable support for **multiple languages**.
- Deploy on **cloud platforms** like Streamlit Cloud or Heroku.
- Implement **user login** and **personalized spam filters**.

---

🔗 Built with ❤️ using [Streamlit](https://streamlit.io/)
