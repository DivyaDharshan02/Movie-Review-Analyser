import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence  # Corrected 'sequance' to 'sequence'
from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import load_model
import streamlit as st


word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

model=load_model("simple_rnn_imdb.h5")

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

def preprocess_text(text):
    words= text.lower().split()
    encoded_review= [word_index.get(word,2) + 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify weather it is positive or Negative")

user_input=st.text_area('movie review')

if st.button('classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment= 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f'sentiment : {sentiment}')
    st.write(f'prediction_score : {prediction[0][0]}')
else:
    st.write('enter a movie review')