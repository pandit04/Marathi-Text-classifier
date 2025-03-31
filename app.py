import streamlit as st
import pickle
import pandas as pd

# Load the vectorizer, model, and label encoder
vectorizer = pickle.load(open('vector.pkl', 'rb'))
nb_model = pickle.load(open('model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Function to preprocess text
def preprocess_text(text):
    marathi_stopwords = set([
        'आणि', 'तो', 'त्यांना', 'यासाठी', 'किंवा', 'हे', 'ही', 'हाच',
        'त्या', 'तिचे', 'पण', 'माझे', 'तुम्ही', 'आहे', 'आहेत', 'असे',
        'तरी', 'असणे', 'जर', 'तुम्ही', 'होईल', 'कस', 'सर्व', 'अनेक',
        'जण', 'अनेक', 'मुळे', 'पुन्हा', 'असताना', 'तसा', 'असले',
        'सुधा', 'जसे', 'नाही', 'ज्या', 'म्हणजे', 'केल्याने', 'तुम्ही',
        'म्हणजे', 'संपूर्ण', 'यावर', 'म्हणजे', 'या', 'त्याचे', 'विषय',
        'माझा', 'जरी', 'आहेत', 'हे', 'तिथे', 'तुम्हाला', 'तर',
        'असेच', 'आणखी', 'मूलतः', 'किंवा', 'रोज', 'त्यांच्यामुळे',
        'वापर', 'असावा', 'वापरून', 'असलेले', 'फार', 'अधूनमधून',
        'असणे', 'गेल्या', 'बद्दल', 'एक', 'तरी', 'कृपया', 'असो',
        'खूप', 'शक्य', 'तुम्ही', 'नाही', 'किंवा', 'याने', 'वर्तमान',
        'सर्वसाधारण', 'आहे', 'अजून', 'अशा', 'तुम्ही', 'अशा',
        'जसा', 'असे', 'ते', 'विनंती', 'आपण', 'येथे', 'तुम्ही',
        'सहा', 'नक्की', 'तुम्ही', 'गेल्यादिवशी', 'ज्यांना', 'त्याचे',
        'म्हणजे', 'आपल्याला', 'त्यांना', 'यावर', 'त्यांना',
        'असलेले', 'आपले', 'म्हणजे', 'वेळी', 'त्यावर', 'तरीही'
    ])
    
    words = [word for word in text.split() if word not in marathi_stopwords]
    return ' '.join(words)

# Function to classify text
def classify_text(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text]).toarray()
    predicted_label = nb_model.predict(text_vector)
    predicted_category = label_encoder.inverse_transform(predicted_label)
    return predicted_category[0]  # Return the category as a string

# Streamlit app layout
st.title("Marathi Text Classification")
st.write("Enter your text below:")

# Text input for user
user_input = st.text_area("Text Input", height=100)

# Button to classify the input text
if st.button("Classify"):
    if user_input:
        predicted_category = classify_text(user_input)
        st.success(f"Predicted Category: {predicted_category}")
    else:
        st.warning("Please enter some text for classification.")
