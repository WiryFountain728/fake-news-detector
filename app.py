# app.py

import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article and I'll tell you if it's **Real or Fake**.")

# Text input
user_input = st.text_area("Paste the news content here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        
        if prediction == 1:
            st.success("✅ This news article is **REAL**.")
        else:
            st.error("⚠️ This news article is **FAKE**.")
