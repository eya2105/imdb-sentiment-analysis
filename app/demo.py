import streamlit as st
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences   
import pickle
import re
import sys

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and the model will predict if it's **positive** or **negative**.")

# Load model and tokenizer with error catching
@st.cache_resource
def load_model():
    try:
        model_path = 'models/bilstm_model.h5' 
        if not tf.io.gfile.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please check the path.")
            return None
        model = tf.keras.models.load_model(model_path, compile=False)
        # Recompile to avoid warnings
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open('models/tokenizer.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

model = load_model()
tokenizer = load_tokenizer()

if model is None or tokenizer is None:
    st.stop()  # Stop execution if essential components missing

MAXLEN = 200   # must match training maxlen

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

review_text = st.text_area("Review:", height=200)

if st.button("Predict"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        try:
            cleaned = preprocess_text(review_text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, padding='post', maxlen=MAXLEN)
            prob = model.predict(padded)[0][0]
            sentiment = "Positive 😊" if prob >= 0.5 else "Negative 😞"
            confidence = prob if prob >= 0.5 else 1 - prob
            st.subheader(f"Prediction: {sentiment}")
            st.write(f"Confidence: {confidence:.4f}")
            st.progress(float(prob))
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")