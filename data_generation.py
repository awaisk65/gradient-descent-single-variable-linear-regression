import numpy as np

np.random.seed(50)
num = 200
ranges = np.array([[1000, 2000], [30, 60]])
x = np.random.uniform(ranges[:, 0], ranges[:, 1], size=(num, ranges.shape[0]))
print(x)
# np.savetxt('housing.csv', x, fmt='%.2f', delimiter=',', header="price,area")
np.savetxt('housing.csv', x, fmt='%.2f', delimiter=',')