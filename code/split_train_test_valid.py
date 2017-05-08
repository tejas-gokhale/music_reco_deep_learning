<<<<<<< HEAD
import random

with open("../data/try.csv") as fr:
    with open("../data/data_train.csv", "w") as f1, open("../data/data_test.csv", "w") as f2, open("../data/data_valid.csv", "w") as f3:
        for line in fr.readlines():
            rd = random.random()
            if rd <= 0.7:
                f = f1
            else:
                if rd <=0.9:
                    f = f2
                else:
                    f = f3
=======
import random

with open("../data/try.csv") as fr:
    with open("../data/data_train.csv", "w") as f1, open("../data/data_test.csv", "w") as f2, open("../data/data_valid.csv", "w") as f3:
        for line in fr.readlines():
            rd = random.random()
            if rd <= 0.7:
                f = f1
            else:
                if rd <=0.9:
                    f = f2
                else:
                    f = f3
>>>>>>> cd32d1e442a63751bf269452cb62c48a8be8bba1
            f.write(line)