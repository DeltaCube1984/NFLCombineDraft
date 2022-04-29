
from numpy.random.mtrand import logistic
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import csv

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt



df = pd.read_csv('Complete_entries_bin.csv')

print(df.head())
print()

X = df.iloc[:,2:11]
y = df['round'].values

print(X.head())
print('\n\n')

categorical_cols = ['Pos']
print(categorical_cols)
print()

X = pd.get_dummies(X)

print('X and y')
print(X.head())
print(y)

for col in X.columns:
    print(col)

print(X.head())




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('post Scaler')


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.10)


print(X_scaled)
clf = LogisticRegression(random_state = 0, solver = 'sag', max_iter = 4000)
model = clf.fit(X_train, y_train)
print('Test')
y_pred = clf.predict(X_test)
print('Logistic Regression Accuracy %s' % accuracy_score(y_pred, y_test))



knn = KNeighborsClassifier(n_neighbors = 5, p = 2)


knn.fit(X_train, y_train)
print('K nearest neighbor Accuracy %s' % knn.score(X_train, y_train))

#knn.fit(X_scaled, y)
#print(knn.score(X_test, y_test))

#y_pred = knn.predict(x_test)
#print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))



#test_df = pd.read_csv('2022.csv')
#X_test = test_df.iloc[:, 2:11]
#print(X_test.shape)

#y_pred = knn.predict(X_test)