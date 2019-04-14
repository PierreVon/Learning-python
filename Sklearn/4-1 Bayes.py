import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

# mul_nb = MultinomialNB()
# mul_nb = BernoulliNB()
mul_nb = GaussianNB()
mul_nb.fit(x_train, y_train)

print(classification_report(mul_nb.predict(x_test), y_test))
print(confusion_matrix(mul_nb.predict(x_test), y_test))