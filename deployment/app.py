from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

app = Flask(__name__, static_folder='static', template_folder='templates')

log_model = load("log_reg_model.pkl")
pipeline = load("pipeline.pkl")
cols = ['LATITUDE','LONGITUDE','ACCLOC','VISIBILITY', 'LIGHT','RDSFCOND' ,'IMPACTYPE','INVTYPE','INVAGE','INJURY', 'DRIVCOND', 'PEDESTRIAN', 'CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH','PASSENGER' ,'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL','DISABILITY' , 'HOOD_158', 'HOOD_140']


@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/result', methods=['POST'])
def result():
    if 'csvfile' not in request.files:
        return 'No file uploaded', 400
    
    csvfile = request.files['csvfile']
    df = pd.read_csv(csvfile)
    
    bool_attributes = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH',
                   'EMERG_VEH', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'PASSENGER']
    
    df[bool_attributes] = df[bool_attributes].fillna("No")
    
    df[bool_attributes] = df[bool_attributes].apply(lambda x: x.map({'Yes': 1, 'No': 0}))
    
    data = df[cols]
    
    data_trans = pipeline.transform(data)
    prediction = log_model.predict(data_trans)
    
    correct = 0
    incorrect = 0
    for i in range (len(prediction)):
        if(prediction[i] == df['ACCLASS'][i]):
            correct += 1
        else:
            incorrect += 1
    
    
    ratio = str(correct) + "/" + str((correct + incorrect))
    accuracy = accuracy_score(df['ACCLASS'], prediction)
    
    
    return render_template('result.html', accuracy = accuracy, ratio = ratio)

if __name__ == '__main__':
    app.run(debug=True)
