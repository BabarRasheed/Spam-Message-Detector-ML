import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transfrom_text(text):
    text = text.lower() # first step
    text = nltk.word_tokenize(text) # 2nd step
    y =[]
    for i in text: # 3rd step
        if i.isalnum():
            y.append(i)
    text = y[:] # step 4th 
    y.clear()
    for i in text: 
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i) 
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return" ".join(y)

tfidf = pickle.load(open('vectorizer.pk1','rb'))
model = pickle.load(open('model.pk1','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message please')
if st.button('Predict'):


    # 1. Preprocess
    transfrom_sms = transfrom_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transfrom_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
