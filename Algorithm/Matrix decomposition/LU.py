from sklearn.datasets import load_iris
import scipy

data = load_iris().data[:3, :3]

lu = scipy.linalg.lu(data, permute_l = True)

print(lu[0])
print(lu[1])

