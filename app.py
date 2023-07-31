from flask import Flask, request, jsonify
import pickle
from collections import OrderedDict

app = Flask(__name__)

def make_tokens(url):
    tokens_by_slash = url.split('/')  # make tokens after splitting by slash
    total_tokens = []
    for token in tokens_by_slash:
        tokens = token.split('-')  # make tokens after splitting by dash
        for sub_token in tokens:
            tkns_by_dot = sub_token.split('.')  # make tokens after splitting by dot
            total_tokens.append(sub_token)  # add token to the list
            total_tokens.extend(tkns_by_dot)  # add dot tokens to the list
    total_tokens = list(OrderedDict.fromkeys(total_tokens))  # remove redundant tokens while preserving order
    total_tokens = [token for token in total_tokens if token != 'com']
    return total_tokens

@app.route('/')
def home():
    return "Hello world, this is a test"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'Invalid data format. Make sure to provide JSON data with a "url" field.'}), 400

    url = data['url']

    # Read the ML model
    try:
        with open("malicious_url_model.pkl", "rb") as file:
            model, vectorizer = pickle.load(file)
    except FileNotFoundError:
        return jsonify({'error': 'Model not found. Make sure the model file "malicious_url_model.pkl" is available.'}), 500
    except Exception as e:
        return jsonify({'error': f'Error loading the model: {str(e)}'}), 500

    # Vectorize the new URL using the loaded vectorizer
    try:
        X_predict = vectorizer.transform([url])
    except Exception as e:
        return jsonify({'error': f'Error vectorizing the URL: {str(e)}'}), 500

    # Make predictions using the loaded model
    try:
        result = model.predict(X_predict)[0]
    except Exception as e:
        return jsonify({'error': f'Error making predictions: {str(e)}'}), 500

    return jsonify({'output': result})

if __name__ == "__main__":
    app.run(debug=True)
