import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt

plt.rc("font", size=14)

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

mydata = pd.read_csv('social.csv',header=0)

mydata = mydata.dropna()

#print(mydata.shape)

#print(list(mydata.columns))

x = mydata.iloc[:, 0:4].values 

y = mydata.iloc[:, 4].values 

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(mydata.head(5))

 

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

result = logreg.fit(x_train, y_train)

print(result)

 

y_pred = logreg.predict(x_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
