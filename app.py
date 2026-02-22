import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import string

ps=PorterStemmer()
def trans(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

model=joblib.load("model.pkl")
vectorizer=joblib.load("vectorizer.pkl")
    
st.title("Spam Finder")

input_data=st.text_area("Enter the message")



if st.button("Predict"):
    transform_data=trans(input_data)
    vectorize_input=vectorizer.transform([transform_data])
    pre=model.predict(vectorize_input)[0]
    if pre==1:
        st.header("SPAM")
    else:
        st.header("HAM")

