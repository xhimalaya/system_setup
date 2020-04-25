while True:
	try:
		import numpy as np
		import matplotlib.pyplot as plt
		import pandas as pd
		import tensorflow as tf
		import seaborn as sns
		break
	except:
		import os
		pkg=["numpy","matplotlib","pandas","theano","keras","tensorflow","seaborn"]
		for i in pkg:
			os.system("python3 -m pip install {}".format(i))


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
print(">"*60)
print(X)
print(">"*60)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50)

y_pred = model.predict(X_test)
print("*"*60)
print(y_pred)
print("*"*60)
y_pred = np.array([np.argmax(i) for i in y_pred])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# accurecy 
from sklearn.metrics import accuracy_score as ac
print(ac(y_test, y_pred))
#heatmap
sns.heatmap(cm)
plt.show()