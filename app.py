from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import OrderedDict
'''
# tokens ending with remove
def makeTokens(f):
    tkns_BySlash = str(f).split('/')  # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # make tokens after splitting by dash
        for token in tokens:
            tkns_ByDot = token.split('.')  # make tokens after splitting by dot
            total_Tokens.append(token)  # add token to the list
            total_Tokens.extend(tkns_ByDot)  # add dot tokens to the list
    total_Tokens = list(OrderedDict.fromkeys(total_Tokens))  # remove redundant tokens while preserving order
    total_Tokens = [token for token in total_Tokens if not any(token.endswith(ext) for ext in ["com"])]
    return total_Tokens
'''
def makeTokens(f):
    tkns_BySlash = str(f).split('/')  # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # make tokens after splitting by dash
        for token in tokens:
            tkns_ByDot = token.split('.')  # make tokens after splitting by dot
            total_Tokens.append(token)  # add token to the list
            total_Tokens.extend(tkns_ByDot)  # add dot tokens to the list
    total_Tokens = list(OrderedDict.fromkeys(total_Tokens))  # remove redundant tokens while preserving order
    for itm in ['com']:
        if itm in total_Tokens:
            total_Tokens.remove(itm)
    return total_Tokens

app=Flask(__name__)

@app.route('/')
def home():
    return "hello world, this is a test"

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
    vectorizer.tokenizer = makeTokens

    # Vectorize the new URLs using the loaded vectorizer
    X_predict = vectorizer.transform(input_query)

    # Make predictions using the loaded model
    result = model.predict(X_predict)[0]
    print(result)

    return jsonify({'output':result})



if __name__ == "__main__":
    app.run(debug=True)
    
