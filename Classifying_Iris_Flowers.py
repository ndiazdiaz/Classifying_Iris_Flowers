# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:19:20 2023

@author: Nerea
"""
#Classifying Iris Flowers

#IMPORTAR LIBRERÍAS
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset
iris_df = pd.read_csv("C:/Users/34677/Downloads/data_flowers.csv", sep=';')
iris_df.sample(frac=1, random_state=seed)

# selecting features and target data
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# save the model to disk
joblib.dump(clf, "rf_model.sav")


import numpy as np

def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)


st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \
     (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Pepal characteristics")
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


st.text('')
st.text('')
st.markdown(
    '`Create by` [santiviquez](https://twitter.com/santiviquez) | \
         `Code:` [GitHub](https://github.com/santiviquez/iris-streamlit)')

