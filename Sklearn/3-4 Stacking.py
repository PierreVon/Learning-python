from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x_data = iris.data[:, 1:3]
y_data = iris.target

clf1 = KNeighborsClassifier()
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()

lr = LogisticRegression()


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)