from flask import Flask, request, jsonify
#import pickle
import numpy as np
import pandas as pd
#import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# url="google.com/search=faizanahmad"
# input_query=np.array([url])
# print(input_query)

# input_query = vectorizer.transform(input_query)
# result = logit.predict(input_query)[0]
# print(result)


app=Flask(__name__)

@app.route('/')
def home():
    return "hello world, this is a test"

@app.route('/predict',methods=['POST'])
def predict():
    url=request.form.get('url')
    #result={'url':url}
    input_query=np.array([url])
    #print(input_query)


    # Reading data from csv file
    data = pd.read_csv('urldata.csv')
    data.head()

    # Labels
    y = data["label"]

    # Features
    url_list = data["url"]


    # Using Tokenizer
    vectorizer = TfidfVectorizer()

    # Store vectors into X variable as Our XFeatures
    X = vectorizer.fit_transform(url_list)



    # Split into training and testing dataset 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Model Building using logistic regression
    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    

    #vectorizer = TfidfVectorizer()
    input_query = vectorizer.transform(input_query)
    result = logit.predict(input_query)[0]
    #print(result)
    return jsonify({'output':result})



if __name__ == "__main__":
    app.run(debug=True)
