from sklearn import svm

x = [[3,3], [4,3], [1,1]]
y = [1, 1, -1]

model = svm.SVC(kernel='linear')
model.fit(x,y)
print(model.support_vectors_)
print(model.support_)
print(model.predict([[4,3]]))
print(model.coef_)
print(model.intercept_)