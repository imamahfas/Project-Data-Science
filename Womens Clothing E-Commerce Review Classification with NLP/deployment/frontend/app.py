import streamlit as st
import requests
import json
import os
import pickle
import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.title('Women Clothing Review Prediction')
with st.form(key='form_parameters'):
    
    review_text = st.text_input('Insert review here')
    submitted = st.form_submit_button('Predict!')


stpwds_en = list(set(stopwords.words('english')))
lemma = WordNetLemmatizer()

def text_proses(teks):
  teks = teks.lower()
  teks = teks.translate(str.maketrans("","",string.punctuation))
  teks = re.sub("@[A-Za-z0-9_]+"," ", teks) #Menghilangkan mention jika ada
  teks = re.sub("#[A-Za-z0-9_]+"," ", teks) #Menghilangkan hashtag jika ada
  teks = re.sub(r"\\n"," ",teks) #Menghilangkan \n jika ada
  teks = teks.strip()
  teks = re.sub(r"http\S+", " ", teks) #Menghilangkan link jika ada
  teks = re.sub(r"www.\S+", " ", teks) #Menghilangkan link jika ada
  teks = re.sub("[^A-Za-z\s']"," ", teks) #Menghilangkan yang bukan huruf jika ada
  tokens = word_tokenize(teks)
  teks = ' '.join([word for word in tokens if word not in stpwds_en])
  teks = ''.join([lemma.lemmatize(words) for words in teks])
  return teks


tx = {'review': [review_text]}
tx_df = pd.DataFrame.from_dict(tx)

# inference
data_new =  tx_df['review'].apply(text_proses)


if submitted:

    input_data_json = json.dumps({
        'signature_name':'serving_default',
        'instances':[[data_new[0]]]
    })

    #URL = "http://127.0.0.1:5000/sales_prediction" # sebelum push backend
    URL = "https://ecommerce-review-imam.herokuapp.com/v1/models/ecommerce_review:predict" # URL Heroku

    # # komunikasi
    r = requests.post(URL, data=input_data_json)
    res = r.json()
    # # st.write(res)
    resultz = res['predictions'][0]
    # st.write(resultz)
    if resultz[0] > 0.5:
         text_hasil = '<h1 style="font-family:Helvetica; color:#8c8c00; text-align:center;">Product Recommended!</h1>'
         st.markdown(text_hasil, unsafe_allow_html=True)
    else:
         text_hasil1 = '<h1 style="font-family:Helvetica; color:#8c8c00; text-align:center;">Product not Recommended!</h1>'
         st.markdown(text_hasil1, unsafe_allow_html=True)
    # if r.status_code == 200:
    #     st.title(res['result']['label_idx'])
    # else:
    #     st.title("ERROR BOSS")
    #     st.write(res['message'])