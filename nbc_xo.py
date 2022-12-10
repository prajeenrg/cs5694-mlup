import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load buys_car dataset discussed in class
df = pd.read_csv("datasets/tic_tac_toe.csv")
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

# split dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.1)

print("\nNo. of training samples:", len(X_train))
print("No. of testing samples:", len(X_test))

# use Naive-Bayes to solve the problem
model = CategoricalNB()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

# show results
print("\nY:", Y_test.to_numpy())
print("Y_predict:", Y_predict)

# show metrics
print("\nTrain Accuracy:", model.score(X_train, Y_train))
print("Test Accuracy:", accuracy_score(Y_test, Y_predict))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_predict))
print("\nClassification Report:\n", classification_report(Y_test, Y_predict))
