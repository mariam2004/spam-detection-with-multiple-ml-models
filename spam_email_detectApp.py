import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image

# Load saved model and vectorizer
model = joblib.load(r"spam_model\best_model_mnb.pkl")
vectorizer = joblib.load(r"spam_model\vectorizer.pkl")

# Sample accuracy data – update these with your actual values
accuracy_data = pd.DataFrame({
    "Model": ["Logistic Regression", "Multinomial NB", "SVM", "Random Forest", "XGBoost"],
    "Train Accuracy": [0.9758, 0.9933, 0.9116, 0.9282, 0.9733],
    "Test Accuracy": [0.9668, 0.9758, 0.9130, 0.9229, 0.9659]
})

# Set up navigation
image = Image.open(r"1694938680443.jpeg")
st.sidebar.image(image, use_container_width=True)

page = st.sidebar.selectbox("📄 Select a Page", ["🧠 Project Overview", "📊 Model Comparison", "📬 Test a Message"])

# ------------------ Page 1: Overview ------------------
if page == "🧠 Project Overview":
    st.title("📩 Spam Message Classifier")
    st.markdown("""
    This project focuses on building a machine learning model to **classify text messages as Spam or Ham** (not spam).

    ### 🔍 Workflow:
    1. **Data Cleaning** – Removing stopwords, lemmatizing, etc.
    2. **Feature Extraction** – Using `CountVectorizer`.
    3. **Model Training** – Multiple ML models were trained.
    4. **Evaluation** – Comparing models on accuracy and generalization.

    ### 🤖 Models Used:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Support Vector Machine (SVM)
    - Random Forest
    - XGBoost
    """)

# ------------------ Page 2: Comparison ------------------
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    
    st.subheader("Train/Test Accuracy Table")
    st.dataframe(accuracy_data)

    # Scatter plot
    st.subheader("📈 Train vs Test Accuracy")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(accuracy_data["Model"], accuracy_data["Train Accuracy"], color="blue", label="Train Accuracy", s=100)
    ax.scatter(accuracy_data["Model"], accuracy_data["Test Accuracy"], color="red", label="Test Accuracy", s=100)
    ax.plot(accuracy_data["Model"], accuracy_data["Train Accuracy"], color="blue", linestyle="--")
    ax.plot(accuracy_data["Model"], accuracy_data["Test Accuracy"], color="red", linestyle="--")
    plt.xticks(rotation=45)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Best model
    best_model = accuracy_data.loc[accuracy_data["Test Accuracy"].idxmax()]
    st.markdown(f"""
    ### 🏆 Best Model:
    **{best_model['Model']}**  
    Test Accuracy: **{best_model['Test Accuracy'] * 100:.2f}%**
    """)

# ------------------ Page 3: Predict ------------------
elif page == "📬 Test a Message":
    st.title("📬 Spam Message Detector")

    st.markdown("Enter a message below to check if it's **Spam** or **Ham** using our best model (Multinomial Naive Bayes):")

    user_input = st.text_area("Type your message here...", height=150)

    if st.button("🔍 Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            # Vectorize input and predict
            user_input_vec = vectorizer.transform([user_input])
            prediction = model.predict(user_input_vec)
            result = "Spam 🚫" if prediction[0] == 1 else "Ham ✅"
            st.success(f"Prediction: **{result}**")
