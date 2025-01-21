# SMS Spam Detection Project

**Overview**

This project is a machine learning application designed to classify SMS or email messages as either Spam or Not Spam. It uses the scikit-learn library for machine learning and the Streamlit framework for creating an interactive web application.

**Features**

Preprocessing of user input (tokenization, stemming, and stopword removal).

Classification of messages using a pre-trained Multinomial Naive Bayes model.

User-friendly interface built with Streamlit.

Real-time prediction results with a visually appealing UI.

**Technologies Used**

Python 3.x

Streamlit for web app deployment

scikit-learn for machine learning

nltk for natural language processing

Pickle for model serialization

**Installation**

Follow these steps to set up and run the project locally:

**Clone the repository:**
```bash
git clone https://github.com/your-repo-name/sms-spam-detection.git
cd sms-spam-detection
```
Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate      # For Windows
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the app:
```bash
streamlit run app.py
```

**How It Works**

Preprocessing:

Converts the input text to lowercase.

Tokenizes the text into words.

Removes non-alphanumeric characters and stopwords.

Applies stemming to reduce words to their root forms.

Vectorization:

The preprocessed text is converted into numerical features using a TF-IDF Vectorizer.

Prediction:

A pre-trained Multinomial Naive Bayes model is used to classify the text as Spam or Not Spam.

Results Display:

The app displays results using color-coded notifications:

Spam: Red alert with an error icon.

Not Spam: Green notification with a success icon.

**Deployment :** https://spamclassifier-fkldr4y4wvfcxjrlmq962e.streamlit.app/
