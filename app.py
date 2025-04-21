import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import spacy
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

lemma = WordNetLemmatizer()
vectorization = TfidfVectorizer()
nlp = spacy.load("en_core_web_sm")

vector_form = pickle.load(open("vector.pkl", "rb"))
load_model = pickle.load(open("model.pkl", "rb"))

list1 = nlp.Defaults.stop_words

list2 = stopwords.words("english")

Stopwords = set((set(list1) | set(list2)))


def clean_text(text):

    string = ""

    # lower casing
    text = text.lower()

    # simplifying text
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)

    # removing any special character
    text = re.sub(r"[-()\"#!@$%^&*{}?.,:]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)

    for word in text.split():
        if word not in Stopwords:
            string += lemma.lemmatize(word) + " "

    return string


def fake_news(news):
    news = clean_text(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


if __name__ == "__main__":
    st.title("Fake News Classification app ")
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class = fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success("Real News")
        if prediction_class == [1]:
            st.warning("Fake News")
