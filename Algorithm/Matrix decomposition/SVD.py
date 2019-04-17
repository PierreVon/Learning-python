import numpy as np
from sklearn.datasets import load_iris

data = load_iris().data[:3, :3]

# A = U*Σ*VT
U, D, VH = np.linalg.svd(data)

print(U, '\n', D, '\n', VH)