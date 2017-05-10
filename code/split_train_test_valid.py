import random

def split_data(file):
    with open(file) as fr:
        with open("../data/new/data_train.csv", "w") as f1, open("../data/new/data_test.csv", "w") as f2, open("../data/new/data_valid.csv", "w") as f3:
            for line in fr.readlines():
                rd = random.random()
                if rd <= 0.4:
                    f = f1
                else:
                    if rd <=0.8:
                        f = f2
                    else:
                        f = f3
                f.write(line)


split_data('../data/inputs.csv')