import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv('4.csv')

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'logistic':make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)),
    'knn':make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5)),
    'mlp' :make_pipeline(StandardScaler(), MLPClassifier(max_iter=10000)),
    'random_forest':make_pipeline(StandardScaler(), RandomForestClassifier()),
}


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train.values, y_train)
    fit_models[algo] = model

print(fit_models['random_forest'].predict(X_test)) 


for algo, model in fit_models.items():
    yhat = model.predict(X_test.values)
    ypred = model.predict(X_train.values)
    cm = confusion_matrix(y_test, yhat)
    

    micro = f1_score(y_test.values, yhat, average='micro')
    macro = f1_score(y_test.values, yhat, average='macro')
    weighted = f1_score(y_test.values, yhat, average='weighted')
    
    print(algo,'accuracy score', accuracy_score(y_test, yhat),)
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


with open('aemn.pkl', 'wb') as f:
#with open('wlasl.pkl', 'wb') as f:
    pickle.dump(fit_models['random_forest'], f)


