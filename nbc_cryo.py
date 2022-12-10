import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load and display dataset
df = pd.read_excel("datasets/Cryotherapy.xlsx")
print("Dataset:")
print(df.head())
print("\nData Description:")
print(df.describe())

# split train and test
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.2)

print("\nAttributes Split-up:")
print("Features:", X.columns.values)
print("Target:", Y.name)

print("\nNo. of training samples:", len(X_train))
print("No. of testing samples:", len(X_test))

# model creation and fitting
model = GaussianNB()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

# show results
print("\nY_test:", Y_test.to_numpy())
print("Y_predict:", Y_predict)

# display metrics
print("\nTraining accuracy:", model.score(X_train, Y_train))
print("Testing accuracy:", accuracy_score(Y_test, Y_predict))
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_predict))
print("\nClassification Report:")
print(classification_report(Y_test, Y_predict))
