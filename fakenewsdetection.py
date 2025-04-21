import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
from sklearn.model_selection import train_test_split

# from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics

nlp = spacy.load("en_core_web_sm")

########## Load the data ##############
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
# print(fake["subject"].value_counts())
# print(true["subject"].value_counts())
fake["category"] = 1
true["category"] = 0
# print(fake.head(5))
# print(fake.tail(5))
# print(true.head(5))
# print(true.tail(5))
df = pd.concat([fake, true]).reset_index(drop=True)
df = df[["text", "category"]]
# print(df.head())
# print(df.tail())

############ Data Cleaning #############
df.isna().sum()
# print(df.isna().sum()*100/len(df))
blanks = []
for index, text in df["text"].items():
    if text.isspace():
        blanks.append(index)

# print(len(blanks))
# print(df.shape)
df.drop(blanks, inplace=True)
# print(df.shape)

lemma = WordNetLemmatizer()

list1 = nlp.Defaults.stop_words
# print(len(list1))
list2 = stopwords.words("english")
# print(len(list2))
Stopwords = set((set(list1) | set(list2)))
# print(len(Stopwords))


# text cleaning function
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


# print(df["text"][1099])
# print("\n\n\n")
# print(clean_text(df["text"][1099]))

df["text"] = df["text"].apply(clean_text)
# print(df["text"])

################## Model Building ###########################
X = df["text"]
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])
# text_clf.fit(X_train, y_train)

vect = TfidfVectorizer()
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)


model = LinearSVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# print(metrics.classification_report(y_test, predictions))
# print(metrics.accuracy_score(y_test, predictions))

pickle.dump(vect, open("vector.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

vector_form = pickle.load(open("vector.pkl", "rb"))
load_model = pickle.load(open("model.pkl", "rb"))


def fake_news(news):
    news = clean_text(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


val = fake_news("""Narendra Modi is prime minister of india. """)
if val == [0]:
    print("Real News")
else:
    print("Fake News")
