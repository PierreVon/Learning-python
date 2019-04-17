from sklearn.datasets import load_iris
import scipy

data = load_iris().data[:3, :3]

cholesky = scipy.linalg.qr(data)
q = cholesky[0]

print('Orthogonal matrix:\n', cholesky[0])
print(cholesky[1])


print('Indentity matrix:\n', scipy.dot(q,scipy.linalg.inv(q)))