import numpy as np
# csv = np.genfromtxt ('../data/all/inputs.csv', delimiter=",")
# print(csv.shape)

from split_train_test_valid import split_data

split_data('../data/all/inputs.csv')