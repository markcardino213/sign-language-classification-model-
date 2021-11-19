import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle 

df = pd.read_csv('coordinates.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



#classifier = LogisticRegression(random_state = 1234, max_iter=10000)
#classifier.fit(X_train.values, y_train)

#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2, max_iter=10000)
#classifier.fit(X_train.values, y_train)

classifier = LinearSVC(max_iter=10000)
classifier.fit(X_train.values, y_train.values)


y_pred = classifier.predict(X_test.values) 


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

print(micro, macro, weighted)



with open('body_language2.pkl', 'wb') as f:
    pickle.dump(classifier, f)