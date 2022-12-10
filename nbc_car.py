import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load buys_car dataset discussed in class
df = pd.read_csv("datasets/buys_car.csv")
print("Dataset:")
print(df.head())

# transform label into numericals
le = LabelEncoder()
for label in df:
    df[label] = le.fit_transform(df[label])
print("\nDataset after encoding:")
print(df.head())
print("\nData Description:")
print(df.describe())

# Split dataset into features and target
print("\nAttributes Split-up:")
print("Features:", df.columns.values[:-1])
print("Target:", df.columns.values[-1])

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

# use Naive-Bayes to solve the problem
model = CategoricalNB()
model.fit(X, Y)
Y_predict = model.predict(X)

# show results
print("\nY:", Y.to_numpy())
print("Y_predict:", Y_predict)

# show metrics
print("\nAccuracy:", accuracy_score(Y, Y_predict))
print("Confusion Matrix:\n", confusion_matrix(Y, Y_predict))
print("\nClassification Report:\n", classification_report(Y, Y_predict))
