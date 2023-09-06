from flask import Flask, request, jsonify
import json
import pickle
import pandas as pd
import NumpyEncoder

app = Flask(__name__)
books = []

@app.route('/api/predict', methods=['POST'])
def predict():
    # Load the pre-trained model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Read data from the POST request
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400
    glucose = [item['Glucose'] for item in data]

    # Perform the prediction
    df = pd.DataFrame(glucose, columns=['Glucose'])
    predictions = model.predict(df)
    encoded_predictions = json.dumps(predictions, cls=NumpyEncoder.NumpyEncoder)

    return jsonify("{predictions:"+encoded_predictions+"}"), 201

if __name__ == '__main__':
    app.run(debug=True)

# curl -X POST -H "Content-Type: application/json" --data @data.json http://localhost:5000/api/predict