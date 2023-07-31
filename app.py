from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import makeTokens


app=Flask(__name__)

@app.route('/')
def home():
    return "hello world, this is a test"
    
vectorizer.tokenizer = makeTokens

@app.route('/predict',methods=['POST'])
def predict():
    url=request.form.get('url')
    #result={'url':url}
    input_query=np.array([url])
    print(input_query)

    # read the ML model
    with open("malicious_url_model.pkl", "rb") as file:
        model, vectorizer = pickle.load(file)

    #X_predict = input_query
    # Re-create the vectorizer object with the same tokenizer function
    

    # Vectorize the new URLs using the loaded vectorizer
    X_predict = vectorizer.transform(input_query)

    # Make predictions using the loaded model
    result = model.predict(X_predict)[0]
    print(result)

    return jsonify({'output':result})



if __name__ == "__main__":
    app.run(debug=True)
    
