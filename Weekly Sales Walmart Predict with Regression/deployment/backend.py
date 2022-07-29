from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Inisiasi 
app = Flask(__name__)

# open model
def open_model(model_path):
    """
    helper function for loading model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model_walmart = open_model("model.pkl")  

# fungsi untuk inference Walmart
def inference_walmart(data, model):
    """
    input : list with length : 4 --> [1, 2, 3, 4]
    output : predicted class (idx, label)
    """
  
    columns = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI','Unemployment', 'Super_Bowl', 'Labor_Day', 'Thanksgiving', 'Christmas']
    data = pd.DataFrame([data], columns=columns)
    res = model.predict(data)
    return res[0]


# halaman home
@app.route("/")
def homepage():
    return "<h1> Deployment Model Backend! </h1>"

# halaman inference Walmart
@app.route('/sales_prediction', methods=['POST'])
def sales_predict():
    """
    content = 
    {
        'Store' : xx,
        'Holiday_Flag' : xx,
        'Temperature' : xx,
        'Fuel_Price' : xx,
        'CPI' : xx,
        'Unemployment' : xx,
        'Super_Bowl' : xx,
        'Labor_Day' : xx,
        'Thanksgiving' : xx,
        'Christmas' : xx,
      
        
    }
    """
    content = request.json
    newdata = [
        content['Store'], 
        content['Holiday_Flag'],
        content['Temperature'],
        content['Fuel_Price'],
        content['CPI'],
        content['Unemployment'],
        content['Super_Bowl'],
        content['Labor_Day'],
        content['Thanksgiving'],
        content['Christmas'],
               ]
    
    res_idx = inference_walmart(newdata, model_walmart)
    result = {
        'label_idx': str(res_idx)
    }
    response = jsonify(success=True,
                       result=result)
    return response, 200

#app.run(debug=True)