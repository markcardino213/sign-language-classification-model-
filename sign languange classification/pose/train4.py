import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle 

df = pd.read_csv('hand_coords2.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

#classifier = make_pipeline(StandardScaler(),MLPClassifier())
#classifier.fit(X_train.values, y_train)

#classifier = make_pipeline(StandardScaler(),DecisionTreeClassifier())
#classifier.fit(X_train.values, y_train)

#classifier = make_pipeline(StandardScaler(),RandomForestClassifier())
#classifier.fit(X_train.values, y_train)

classifier =  LogisticRegression(max_iter=10000)
classifier.fit(X_train.values, y_train)

#classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
#classifier.fit(X_train.values, y_train)

#classifier = make_pipeline(StandardScaler(),KNeighborsClassifier())
#classifier.fit(X_train.values, y_train)

#classifier = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
#classifier.fit(X_train.values, y_train.values)

#classifier = make_pipeline(StandardScaler(), SVC(max_iter=10000))
#classifier.fit(X_train.values, y_train.values)

y_pred = classifier.predict(X_test.values) 
y_pred2 = classifier.predict(X_train.values)

#print(classifier.predict(X_test.values))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.values, y_pred)

TP = cm[0][0]
FP = cm[0][1]
TN = cm[1][1]
FN = cm[1][0]
N = (TP + FP) + (TN + FN)
#or 
#N = (TP + FN) + (FP + TN)
#where N is the total Population
accuracy = (TP + FP) / N
print(round(accuracy, 2))

from sklearn.metrics import f1_score
micro = f1_score(y_test.values, y_pred, average='micro')
macro = f1_score(y_test.values, y_pred, average='macro')
weighted = f1_score(y_test.values, y_pred, average='weighted')

print("test score")
print(micro, macro, weighted)

micro2 = f1_score(y_train.values, y_pred2, average='micro')
macro2 = f1_score(y_train.values, y_pred2, average='macro')
weighted2 = f1_score(y_train.values, y_pred2, average='weighted')

print("train score")
print(micro2, macro2, weighted2)


with open('signlanguage.pkl', 'wb') as f:
    pickle.dump(classifier, f)