import numpy as np
grad = np.zeros(5)
grad *= 2/1
grad += np.ones(5) * 5
print(grad)

def lossGrad(data, labels, weights, numDataPoint, numFeat):
    #grad = np.zeros(numFeat)
    print("res", data[5] * (np.dot(data[5], np.transpose(weights)) - labels[5]) 
  # for index in range(numDataPoint):
    #     #print(np.dot(data[index], np.transpose(weights)) - labels[index])
    #     grad += (np.dot(data[index], np.transpose(weights))[0] - labels[index]) * data[index]
    #grad *= 2/numDataPoint
    return grad  