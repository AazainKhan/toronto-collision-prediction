from flask import *
import numpy as np
import pandas as pd
from joblib import load

app = Flask(__name__)

log_model = load("log_reg_model.pkl")
pipeline = load("pipeline.pkl")
cols = ['VISIBILITY', 'LIGHT', 'INJURY', 'DRIVCOND', 'PEDESTRIAN', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'HOOD_158', 'HOOD_140']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    VISIBILITY = np.array([request.form['VISIBILITY']])
    LIGHT = np.array([request.form['LIGHT']])
    INJURY = np.array([request.form['INJURY']])
    DRIVCOND = np.array([request.form['DRIVCOND']])
    PEDESTRIAN = np.array([request.form['PEDESTRIAN']])
    TRUCK = np.array([request.form['TRUCK']])
    TRSN_CITY_VEH = np.array([request.form['TRSN_CITY_VEH']])
    EMERG_VEH = np.array([request.form['EMERG_VEH']])
    SPEEDING = np.array([request.form['SPEEDING']])
    AG_DRIV = np.array([request.form['AG_DRIV']])
    REDLIGHT = np.array([request.form['REDLIGHT']])
    ALCOHOL = np.array([request.form['ALCOHOL']])
    HOOD_158 = np.array([request.form['HOOD_158']])
    HOOD_140 = np.array([request.form['HOOD_140']])
    final = np.concatenate([VISIBILITY, LIGHT, INJURY, DRIVCOND, PEDESTRIAN, TRUCK, TRSN_CITY_VEH, EMERG_VEH, SPEEDING, AG_DRIV, REDLIGHT, ALCOHOL, HOOD_158, HOOD_140])
    final = np.array(final)
    data = pd.DataFrame([final], columns=cols)
    data_trans = pipeline.transform(data)
    prediction = log_model.predict(data_trans)
    return render_template('result.html', prediction = prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
