import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# load data and display
df = pd.read_csv("datasets/forest_fires.csv")
print("Dataset:")
print(df.head())
print("\nDataset Decription")
print(df.describe())
print("\nAttributes Split-up:")
print("Features:", df.columns.values[:-1])
print("Target:", df.columns.values[-1])

# scaling and labelling features
ss = StandardScaler()
le = LabelEncoder()
df.iloc[:,:-1] = ss.fit_transform(df.iloc[:,:-1])
df.iloc[:,-1] = le.fit_transform(df.iloc[:,-1])
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

print("\nData after transform and labelling:")
print(df.head())

# splitting dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.15)

print("\nNo. of training samples:", len(X_train))
print("No. of testing samples:", len(X_test))

# model MLP and test
model = MLPClassifier(hidden_layer_sizes=(6,), activation='tanh', random_state=1, learning_rate_init=0.3)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)

# print results
print("\nY_test:", Y_test.to_numpy())
print("Y_predict:", Y_predict)

# print metrics
print("\nTraining Accuracy:", model.score(X_train, Y_train))
print("Testing Accuracy:", accuracy_score(Y_test, Y_predict))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_predict))
print("\nClassification Report:")
print(classification_report(Y_test, Y_predict))

# model attributes
print("\nModel Coefficients:")
print(model.coefs_)
print("Model biases:", model.intercepts_)

# plot loss curve
plt.plot(model.loss_curve_)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss')
plt.show()
