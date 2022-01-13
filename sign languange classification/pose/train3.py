import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv('pose_coords.csv')

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)),
    'knn':make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5)),
    'dt':make_pipeline(StandardScaler(), DecisionTreeClassifier()),
    'svc':make_pipeline(StandardScaler(), SVC(max_iter=10000)),
    'lsvc':make_pipeline(StandardScaler(), LinearSVC(max_iter=10000)),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
}


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train.values, y_train)
    fit_models[algo] = model


for algo, model in fit_models.items():
    yhat = model.predict(X_test.values)
    ypred = model.predict(X_train.values)
    cm = confusion_matrix(y_test, yhat)

    micro = f1_score(y_test.values, yhat, average='micro')
    macro = f1_score(y_test.values, yhat, average='macro')
    weighted = f1_score(y_test.values, yhat, average='weighted')
    
    print(algo,'test score', accuracy_score(y_test, yhat),)
    TP = cm[0][0]
    FP = cm[0][1]
    TN = cm[1][1]
    FN = cm[1][0]
    P = TP/(TP + FP)

    R = TP/(TP + FN)

    N = (TP + FP) + (TN + FN)
    F = (P*R)/(P+R)
    F1 = (2*P*R)/(P+R)

    #print(micro, macro, weighted)



with open('pose2.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)


