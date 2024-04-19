import logging
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and preprocessing pipeline
logging.info("Loading model and pipeline...")
model = joblib.load('log_reg_model.pkl')
pipeline = joblib.load('log_reg_pipeline.pkl')
logging.info("Model and pipeline loaded successfully.")

# Load the trained model and preprocessing pipeline
model = joblib.load('log_reg_model.pkl')
pipeline = joblib.load('log_reg_pipeline.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form_data = request.form.to_dict()

    # Print form data for debugging
    app.logger.debug("Form data: %s", form_data)

    # Preprocess form data using the pipeline
    input_data = np.array(list(form_data.values())).reshape(1, -1)
    
    # Debugging statements for ColumnTransformer
    app.logger.debug("Input data shape before transformation: %s", input_data.shape)
    app.logger.debug("Columns before transformation: %s", pipeline.named_steps['preprocessor'].transformers_)

    processed_input = pipeline.transform(input_data)

    # Make prediction
    prediction = model.predict(processed_input)

    # Interpret prediction result
    if prediction[0] == 'Fatal':
        result = 'Fatal'
    else:
        result = 'Not Fatal'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
