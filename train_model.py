import pandas as pd
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download("stopwords")

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove empty rows
data = data.dropna()

# Stopwords
stop_words = set(stopwords.words("english"))

# Clean text
def clean_text(text):
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

# Convert labels
data["label"] = data["label"].map({
    "FAKE": 0,
    "REAL": 1
})

# Input and output
X = data["text"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy:", accuracy * 100, "%")

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")
