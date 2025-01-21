"""import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from sklearn.utils.validation import check_is_fitted
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # converted to a list

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]  # to clone list
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
        """
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from sklearn.utils.validation import check_is_fitted
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load saved models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# App Title and Description
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📩 Email/SMS Spam Classifier")
st.markdown("""
This app uses a **machine learning model** to classify whether a given email or SMS is **Spam** or **Not Spam**.  
Simply type your message below, and the app will predict its category.
""")

# Input Section
st.write("### Input the message you'd like to analyze:")
input_sms = st.text_area("💬 Enter the message:", height=150)

# Predict Button
if st.button('🔍 Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display Result
    if result == 1:
        st.error("🚨 **This message is classified as Spam.**")
    else:
        st.success("✅ **This message is Not Spam.**")

# Footer
st.markdown("""
---
👨‍💻 Developed by [Vaishnavie Tarekar](#)  
📖 Powered by **Streamlit** | Machine Learning with **scikit-learn**
""")

