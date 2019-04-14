from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report

iris = load_iris()

x_data = iris.data
y_data = iris.target

model = tree.DecisionTreeClassifier()
model.fit(x_data, y_data)

print(classification_report(y_data, model.predict(x_data)))