import numpy as np
from random import random, randrange


def sigmoid(w, b, x):
    return 1 / (1 + np.exp(-(b + w*x)))


def create_data(x, y):
    for i in range(149):
        x.append(x[i] + 4 * random() - 1)
        y.append(np.random.uniform(0, 1))
    return x, y


def loss_cal(w, b, x, y):
    loss = 0
    for i in range(100):
        if sigmoid(w, b, x[i]) == 0:
            loss += y[i] * np.log(sigmoid(w, b, x[i])) + (1 - y[i])*np.log(1 - sigmoid(w, b, x[i]))
    return loss/100


def log_reg(w, b, x, y):
    lr_w = 0.0001
    lr_b = 0.01
    for i in range(200000):
        dw = 0
        db = 0
        for j in range(11):
            dw += x[j] * (sigmoid(w, b, x[j]) - y[j])/11
            db += (sigmoid(w, b, x[j]) - y[j])/11
        w -= dw*lr_w
        b -= db*lr_b
    return w, b


x = [0,1,2,3,4,5,6,7,8,9,10]
y = [0,0,0,0,1,0,0,1,0,1,1]
w, b = log_reg(0, 0, x, y)
print(str(w) + " " + str(b))