import numpy as np
csv = np.genfromtxt ('../data/inputs.csv', delimiter=",")
print(csv.shape)

from split_train_test_valid import split_data

split_data('../data/inputs.csv')