import streamlit as st
import pickle
import re
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")


# -----------------------------
# Get current directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Load Model
# -----------------------------
model_path = r"G:\DS_Projects\.venv\AI Echo\model\lstm_sentiment_model.keras"
model = load_model(model_path)


# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)


# -----------------------------
# Load Label Encoder
# -----------------------------
label_encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

with open(label_encoder_path, "rb") as f:
    le = pickle.load(f)


# -----------------------------
# Stopwords + Lemmatizer
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(seq, maxlen=100)

    prediction = model.predict(padded)

    predicted_class = prediction.argmax(axis=1)

    sentiment = le.inverse_transform(predicted_class)

    return sentiment[0]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Echo Sentiment Analyzer")

st.title("🤖 AI Echo: ChatGPT Review Sentiment Analysis")

st.write("Enter a review and the model will predict whether the sentiment is **Positive, Neutral, or Negative**.")


# Text input
user_input = st.text_area("Enter your review here")


# Button
if st.button("Analyze Sentiment"):

    if user_input.strip() != "":

        result = predict_sentiment(user_input)

        if result == "Positive":
            st.success(f"Predicted Sentiment: {result} 😊")

        elif result == "Neutral":
            st.info(f"Predicted Sentiment: {result} 😐")

        else:
            st.error(f"Predicted Sentiment: {result} 😠")

    else:
        st.warning("Please enter a review before clicking the button.")