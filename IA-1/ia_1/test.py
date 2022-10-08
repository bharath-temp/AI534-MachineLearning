import numpy as np
a = np.array([[5., 9., 13], [14., 10., 12.], [11., 15., 19.]])
meanVal = np.mean(a, dtype=np.float64, axis=0)
stdVal = np.std(a, dtype=np.float64, axis=0)
print(stdVal)
for col in range(3):
    a[:,col] /= meanVal[col]
print(a)
