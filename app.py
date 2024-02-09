# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 15:20:48 2023

@author: Punam
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")

# Load the saved model
model_file_path = "D:/Job_ready/projects/Heart_failure_prediction/Heartfailure-Prediction-main/trained_model.sav"

try:
    loaded_model = pickle.load(open(model_file_path, 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    loaded_model = None

# Create a function for prediction
def heartfailure_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Add this line to check the type of loaded_model
    print(type(loaded_model))
    
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'serum_creatinine': float(request.form['serum_creatinine']),
            'age': int(request.form['age']),
            'cp': int(request.form['cp']),
            'high_blood_pressure': int(request.form['high_blood_pressure']),
            'anaemia': int(request.form['anaemia']),
            'creatinine_phosphokinase': int(request.form['creatinine_phosphokinase']),
            'Cholesterol': int(request.form['Cholesterol']),
            'Weight': float(request.form['Weight'])
        }

        result = heartfailure_prediction(list(input_data.values()))

        if result == 0:
            diagnosis = 'The person is not heart failure'
        else:
            diagnosis = 'The person is heart failure'

        return render_template('result.html', diagnosis=diagnosis)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
