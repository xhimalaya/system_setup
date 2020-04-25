import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv('capture_data.csv')
X = dataset.iloc[:, :5].values
y = dataset.iloc[:, 5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 2] = labelencoder_X_3.fit_transform(X[:, 2])
labelencoder_X_4 = LabelEncoder()
X[:, 3] = labelencoder_X_4.fit_transform(X[:, 3])
X = X.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier as rg
rg1=rg(n_estimators=1000,criterion="entropy",n_jobs=10000)
rg1.fit(X_train,y_train)
y_pred=rg1.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)
from sklearn.metrics import confusion_matrix as cm
cm1=cm(y_test,y_pred)
print(cm1)
sns.heatmap(cm1)