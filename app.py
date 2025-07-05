# app.py

import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only the necessary (safe) NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_input_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    tokens = text.split()                # ✅ Simple tokenization (no punkt!)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Enter a news article below and the model will classify it as **Real** or **Fake** news.")

# User input
user_input = st.text_area("Paste the news content here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Clean and predict
        cleaned_text = clean_input_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text)[0][1]

        confidence = model.predict_proba(vectorized_text)[0][1]

# 🔧 Set custom threshold (tune this based on your model)
threshold = 0.15

if confidence >= threshold:
    st.success("✅ This news article is **REAL**.")
else:
    st.error("🚨 This news article is **FAKE**.")

st.write(f"🧠 Model confidence it's real: **{confidence * 100:.2f}%**")
