import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, app, jsonify, render_template, request, url_for

app = Flask(__name__, template_folder = 'templates')
# Load the model

with open('regmodel.pkl', 'rb') as file:
    regmodel = pickle.load(file, encoding='utf-8')

with open('scaling.pkl', 'rb') as file:
    scaler = pickle.load(file, encoding='utf-8')

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values()))).reshape(1,-1)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict", methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The house price prediction is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)

