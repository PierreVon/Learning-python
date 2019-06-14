from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report
import graphviz

iris = load_iris()

x_data = iris.data
y_data = iris.target

model = tree.DecisionTreeClassifier()
model.fit(x_data, y_data)

dot_data = tree.export_graphviz(model,out_file='tree.dot',
                                    feature_names=['SepaLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
                                    class_names=['setosa', 'versicolor', 'virginica'],
                                    filled=True, rounded=True,
                                    special_characters=True)
graph = graphviz.Source(dot_data)

print(classification_report(y_data, model.predict(x_data)))