# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17NZ_7ngdDzGjKR7MeH0hFO3lXveEvVdZ
"""

# Import our libraries

import statsmodels.formula.api as smf
import pandas as pd

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data.head()

model = smf.ols("trips ~ hour", data=data)

modelFit = model.fit()

modelFit.summary()

test_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")

X_test = test_data[['hour']]

pred = modelFit.predict(X_test)