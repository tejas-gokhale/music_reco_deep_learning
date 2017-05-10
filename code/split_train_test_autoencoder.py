import random

def split_data(file):
    with open(file) as fr:
        with open("../data/new/data_train_ae.csv", "w") as f1, open("../data/new/data_test_ae.csv", "w") as f2:
            for line in fr.readlines():
                rd = random.random()
                if rd <= 0.5:
                    f = f1
                else:
                    f = f2
                f.write(line)


split_data('../data/inputs.csv')