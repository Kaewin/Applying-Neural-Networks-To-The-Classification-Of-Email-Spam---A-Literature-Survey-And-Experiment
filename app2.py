import io
import sys
import nltk
from nltk.corpus import stopwords
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.metrics import Precision
import email
from email.message import EmailMessage

def load_model():
    model = model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.load_weights('model.h5')

    return model

def get_raw_text():
    """Prompts the user to input the copied raw text of an email."""

    st.write("Please copy and paste the raw text of an email:")

    line = st.text_input("")

    if line.strip():  # Check if the input is not empty or only contains whitespace
        email_text = line + "\n"
    else:
        email_text = ""

    return email_text

def get_eml_text(uploaded_file):
    eml_content = uploaded_file.getvalue().decode('utf-8')
    return eml_content

def extract_features(email):
    features = {
        'number_of_words': len(email.split()),
        'number_of_stop_words': len([word for word in email.split() if word in list(nltk.corpus.stopwords.words('english'))]),
        'number_of_unique_words': len(set(email.split())),
        'ratio_of_lowercase_to_uppercase': float(len([word for word in email.split() if word.islower()])) / len(email.split()) if len(email.split()) else 0,
        'number_of_exclamation_points': email.count('!'),
    }

    # Tokenize the message body
    tokens = nltk.word_tokenize(email)

    # Remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]

    # Stem the tokens
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Lemmatize the tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    features['number_of_unique_stemmed_words'] = len(set(stemmed_tokens))
    features['number_of_lemmatized_words'] = len(set(lemmatized_tokens))

    cleaned_body = ' '.join([stemmer.stem(token) for token in email.split() if token not in stop_words])
    print('in features function', features)
    return features

def main():

    model = load_model()

    st.write("The model has been loaded.")

    st.write("Welcome To My Email Spam Classifier!")

    st.write("My model takes .eml files, or raw text. Which would you like to submit?")

    options = [".eml file", "raw text"]

    choice = st.selectbox("Select an option", options)

    if choice == ".eml file":
        uploaded_file = st.file_uploader("Upload an .eml file")
        if uploaded_file is not None:
            email_text = get_eml_text(uploaded_file)
            features = extract_features(email_text)
            # turn the features into a dataframe
            features_df = pd.DataFrame(features, index=[0])
            # print(features_df)
            prediction = model.predict(features_df)
            
            # if prediction < 5, return "Spam"
            # if prediction > 5, return "Not Spam"
            if prediction > 0.5:
                prediction = "Spam"
            else:
                prediction = "Not Spam"
            
            st.write("The prediction is:")
            st.write(prediction)

    elif choice == "raw text":
        email_text = get_raw_text()

        if email_text.strip():  # Check if email_text is not empty or only contains whitespace
            features = extract_features(email_text)
            # turn the features into a dataframe
            features_df = pd.DataFrame(features, index=[0])
            print(features_df)
            prediction = model.predict(features_df)
            
            # if prediction < 5, return "Spam"
            # if prediction > 5, return "Not Spam"
            if prediction > 0.5:
                prediction = "Spam"
            else:
                prediction = "Not Spam"
            
            st.write("The prediction is:")
            st.write(prediction)

    else:
        pass

if __name__ == '__main__':
    main()