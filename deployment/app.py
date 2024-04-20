from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

log_model = joblib.load('log_reg_model.pkl')
print('Loaded logistic regression model from log_reg_model.pkl')

svm_model = joblib.load('svm_model.pkl')
print('Loaded SVM model from svm_model.pkl')

nn_model = joblib.load('nn_model.pkl')
print('Loaded neural networks model from nn_model.pkl')

dt_model = joblib.load('dt_model.pkl')
print('Loaded decision tree model from dt_model.pkl')

pipeline = joblib.load('pipeline.pkl')
print('Loaded pipeline from pipeline.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        data = {k: int(v) if v.isdigit() else v for k, v in data.items()}

        # Get the model name from the request
        model_name = data.pop('model')

        data = pd.DataFrame([data])
        data = pipeline.transform(data)

        # Use the appropriate model for prediction
        if model_name == 'logistic_regression':
            prediction = log_model.predict(data)[0]
            probabilities = log_model.predict_proba(data)[0]
            print(f'Used logistic regression model from log_reg_model.pkl for prediction')
        elif model_name == 'svm':
            prediction = svm_model.predict(data)[0]
            probabilities = svm_model.predict_proba(data)[0]
            print(f'Used SVM model from svm_model.pkl for prediction')
        elif model_name == 'neural_networks':
            prediction = nn_model.predict(data)[0]
            probabilities = nn_model.predict_proba(data)[0]
            print(f'Used neural networks model from nn_model.pkl for prediction')
        elif model_name == 'decision_trees':
            prediction = dt_model.predict(data)[0]
            probabilities = dt_model.predict_proba(data)[0]
            print(f'Used decision tree model from dt_model.pkl for prediction')
        else:
            return jsonify({'error': 'Invalid model selected'}), 400

        return jsonify({'result': prediction, 'probabilities': probabilities.tolist()})
    return render_template('index.html')


@app.route('/eda' , methods=['GET', 'POST'])
def eda():
    if request.method == 'POST':
        data = request.form.to_dict()
        data = {k: int(v) if v.isdigit() else v for k, v in data.items()}
        data = pd.DataFrame([data])
        return render_template('eda.html', data=data.to_html())
    return render_template('eda.html')

if __name__ == '__main__':
    app.run(debug=True)
