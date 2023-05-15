import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

df = pd.read_csv("data.csv")

from sklearn.model_selection import train_test_split

X, y = df[["R&D Spend", "Administration", "Marketing Spend"]], df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state= 42)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(clf, pickle_out)
pickle_out.close()