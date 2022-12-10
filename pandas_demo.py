import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Read data from csv
df1 = pd.read_csv("datasets/tic_tac_toe.csv")
print("df1.head():\n", df1.head())
print("\ndf1.tail():\n", df1.tail())

# Read data from .xlxs file
df2 = pd.read_excel("datasets/Cryotherapy.xlsx")
print("\ndf2.head(3):\n", df2.head(3))
print("\ndf2.tail(4):\n", df2.tail(4))

# describe dataframe
print("\ndf1.describe():\n", df1.describe())

# info of dataframe
print()
df2.info()

# decribe all types except int64
print("\ndf2.describe(exclude='int64'):\n", df2.describe(exclude="int64"))

# list unique count
print("\ndf1.nuinque():\n", df1.nunique())

# count values
print("\ndf2.sex.value_counts():\n", df2.sex.value_counts())

# groups
dfgroup = df2.groupby("Result_of_Treatment")
print("\ndfgroup.ngroups:", dfgroup.ngroups)
print("\ndfgroup.groups:\n", dfgroup.groups)
print("\ndfgroup.size():\n", dfgroup.size())
print("\ndfgroup.mean():\n", dfgroup.mean())
print("\ndfgroup.count():\n", dfgroup.count())
print("\ndfgroup.max():\n", dfgroup.max())
print("\ndfgroup.min():\n", dfgroup.min())
print("\ndfgroup.median():\n", dfgroup.median())
print("\ndfgroup.agg(['max', 'min', 'mean']):\n", dfgroup.agg(['max', 'min', 'mean']))
dfgroup.mean().plot.bar()
plt.show()
