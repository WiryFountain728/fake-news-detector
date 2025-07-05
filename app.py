# app.py

import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_input_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove numbers
    tokens = text.split()                # simple tokenization
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Enter a news article and the model will classify it as **Real** or **Fake** news.")

# Text input
user_input = st.text_area("Paste the news content here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned_text = clean_input_text(user_input)
        if cleaned_text.strip() == "":
            st.error("🛑 The cleaned text is empty after preprocessing. Try more informative text.")
        else:
            vectorized_text = vectorizer.transform([cleaned_text])
            try:
                confidence = model.predict_proba(vectorized_text)[0][1]
            except AttributeError:
                st.error("❌ The loaded model doesn't support `predict_proba()`. Use Logistic Regression, Random Forest, etc.")
                confidence = 0.5  # fallback

            # Adjust the threshold as needed
            threshold = 0.15

            if confidence >= threshold:
                st.success("✅ This news article is **REAL**.")
            else:
                st.error("🚨 This news article is **FAKE**.")

            st.write(f"🧠 Confidence it's real: **{confidence * 100:.2f}%**")
