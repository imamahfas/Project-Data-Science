import streamlit as st
import requests
import json
import os
import pickle
import pandas as pd



base_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_path, 'model')
prepro_name = 'preprocessor.pkl'
prepro_path = os.path.join(model_path, prepro_name)
with open(prepro_path, "rb") as filename:
    print(filename)
    preprocessor = pickle.load(filename)
    
prepro_catoh = 'categorical_onehot_pipeline.pkl'
prepro_catoh_path = os.path.join(model_path, prepro_catoh)
with open(prepro_catoh_path, "rb") as filename:
    print(filename)
    categorical_onehot_pipeline = pickle.load(filename)
    
prepro_catord = 'categorical_ord_pipeline.pkl'
prepro_catord_path = os.path.join(model_path, prepro_catord)
with open(prepro_catord_path, "rb") as filename:
    print(filename)
    categorical_ord_pipeline = pickle.load(filename)
    
numerical_norm = 'numerical_norm_pipeline.pkl'
numerical_norm_filepath = os.path.join(model_path, numerical_norm)
with open(numerical_norm_filepath, "rb") as filename:
    numerical_norm_pipeline = pickle.load(filename)
    
numerical_skew = 'numerical_skew_pipeline.pkl'
numerical_skew_path = os.path.join(model_path, numerical_skew)
with open(numerical_skew_path, "rb") as filename:
    print(filename)
    numerical_skew_pipeline = pickle.load(filename)

st.title('Telco Churn Prediction')
tenure = st.number_input('insert tenure')
MonthlyCharges = st.number_input('Insert monthly charges')
TotalCharges = st.number_input("Insert total charges")
MultipleLines = st.selectbox("Multiplelines or not", ['No phone service', 'No', 'Yes'])
InternetService = st.selectbox("Internet service type", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online security or not", ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox("Online backup or not", ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox("Tech support or not", ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox("Streaming movies or not", ['Yes', 'No', 'No internet service'])
Dependents  = st.selectbox("She/He has dependencies?", ['Yes', 'No'])
PhoneService = st.selectbox("Phone service or not", ['Yes', 'No'])
PaperlessBilling = st.selectbox("PaperlessBilling", ['Yes', 'No'])
Contract = st.selectbox("Choose Contract", ['Month-to-month', 'One year', 'Two year'])
SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
st.caption('0 for under 65, 1 for 65 above')


# inference
data = {'tenure':[tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'TechSupport': [TechSupport],
        'StreamingMovies': [StreamingMovies],
        'Dependents': [Dependents],
        'PhoneService': [PhoneService],
        'PaperlessBilling': [PaperlessBilling],
        'Contract': [Contract],
        'SeniorCitizen': [SeniorCitizen]
        }

prepro = pd.DataFrame.from_dict(data)
prepro2 = preprocessor.transform(prepro)
prepro2 = prepro2.tolist()


input_data_json = json.dumps({
    'signature_name':'serving_default',
    'instances':prepro2
})

#URL = "http://127.0.0.1:5000/sales_prediction" # sebelum push backend
URL = "https://telco-churn-imam.herokuapp.com/v1/models/telco_churn:predict" # URL Heroku

# komunikasi
r = requests.post(URL, data=input_data_json)
res = r.json()
# st.write(res)
resultz = res['predictions'][0]
if resultz[0] > 0.5:
    text_hasil = '<h1 style="font-family:Helvetica; color:#8c8c00; text-align:center;">He/She will Churn! not loyal customer</h1>'
    st.markdown(text_hasil, unsafe_allow_html=True)
else:
    text_hasil1 = '<h1 style="font-family:Helvetica; color:#8c8c00; text-align:center;">He/She is loyal customer!</h1>'
    st.markdown(text_hasil1, unsafe_allow_html=True)
# if r.status_code == 200:
#     st.title(res['result']['label_idx'])
# else:
#     st.title("ERROR BOSS")
#     st.write(res['message'])