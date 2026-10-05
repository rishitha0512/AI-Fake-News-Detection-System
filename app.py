import streamlit as st
import joblib
import re

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("📰 AI-Based Fake News Detection System")

st.write(
    "Enter a news article or headline to predict whether it is likely to be REAL or FAKE."
)

news = st.text_area(
    "Enter News:",
    height=200,
    placeholder="Type or paste news here..."
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter some news.")

    else:
        cleaned_news = clean_text(news)

        news_vector = vectorizer.transform([cleaned_news])

        result = model.predict(news_vector)[0]

        probability = model.predict_proba(news_vector)[0]

        if result == 1:
            st.success("✅ Prediction: REAL NEWS")
            st.write(f"Confidence: {probability[1] * 100:.2f}%")

        else:
            st.error("❌ Prediction: FAKE NEWS")
            st.write(f"Confidence: {probability[0] * 100:.2f}%")
