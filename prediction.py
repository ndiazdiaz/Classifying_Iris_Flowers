# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:16:49 2023

@author: Nerea
"""

import joblib


def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)