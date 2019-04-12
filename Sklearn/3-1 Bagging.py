from sklearn import neighbors
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x_data = iris.data[:, :2]
y_data = iris.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)


def plot(model):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, z)


knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)
print("Knn's accuracy is: ", knn.score(x_test, y_test))

dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)
print("Decision tree's accuracy is: ", dtree.score(x_test, y_test))

# Bagging
# randomly extract same amount of subsets(m) from original set
# using SAME algorithm trains all subsets then we get m classifiers
# using m classifiers to predict test data then we will get m answers among all classes
# voting all answers, the maximum number of class would be the final class
bagging_knn = BaggingClassifier(knn, n_estimators=50)
bagging_knn.fit(x_train, y_train)
print("Bagging knn's accuracy is: ", bagging_knn.score(x_test, y_test))

bagging_tree = BaggingClassifier(knn, n_estimators=50)
bagging_tree.fit(x_train, y_train)
print("Bagging tree's accuracy is: ", bagging_tree.score(x_test, y_test))

plt.subplot(221)
plot(knn)
plt.title('KNN')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.subplot(222)
plot(dtree)
plt.title('Decision Tree')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.subplot(223)
plot(bagging_knn)
plt.title('Bagging KNN')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.subplot(224)
plot(bagging_tree)
plt.title('Bagging Tree')
plt.scatter(x_data[:, 0], x_data[:, 1], c = y_data)
plt.show()