from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv

Dtree = open(r'data/buyHouse.csv', 'r')
reader = csv.reader(Dtree)
headers = reader.__next__()
print(headers)

featureList = []
labelList = []

# extract data and labels
for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# change data format to 01
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()
print('x_data:' + str(x_data))

# print label, header of x_data
print(vec.get_feature_names())
print('labelList:' + str(labelList))

lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print('y_data: ' + str(y_data))

model = tree.DecisionTreeClassifier(criterion='entropy')
# model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(x_data, y_data)

x_test = x_data[0]
print('prediction is: ', model.predict(x_test.reshape(1,-1)))