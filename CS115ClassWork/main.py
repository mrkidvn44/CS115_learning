from random import random, randrange
from tabulate import tabulate


def loss_cal(W1, W2, B, X1, X2, Y):
  Loss = 0
  for k in range(100):
    Loss += ((Y[k] - W1*X1[k] -W2*X2[k] - B)**2)
  return Loss/10


if __name__ == '__main__':
    x1 = [10]
    x2 = [5]
    y = [42]
    for i in range(149):
        x1.append(x1[i] + 4*random() - 1)
        x2.append(x2[i] + 4*random() - 1)
        y.append(x1[i+1]*2 + 3*x2[i+1] + 7 + randrange(-1, 3, 2))
    w1, w2, b = 0, 0, 0
    learning_rate = 0.000001
    learning_rateB = 0.01
    loss = loss_cal(w1, w2, b, x1, x2, y)
    for j in range(100000):
        dw1_loss, dw2_loss, db_loss = 0, 0, 0
        for i in range(100):
            dw1_loss += -2 * x1[i] * (y[i] - w1 * x1[i] - w2 * x2[i] - b)
            dw2_loss += -2 * x2[i] * (y[i] - w1 * x1[i] - w2 * x2[i] - b)
            db_loss += -2 * (y[i] - w1 * x1[i] - w2 * x2[i] - b)
        w1 = w1 - learning_rate * dw1_loss/100
        w2 = w2 - learning_rate * dw2_loss/100
        b = b - learning_rateB * db_loss/100
    print(str(w1) + " " + str(w2) + " " + str(b))
    loss = 0
    log_y = []
    for i in range(100, 150):
        log_y.append(x1[i]*w1 + x2[i]*w2 + b)
        loss += (y[i] - log_y[i-100])**2
    loss = loss/50
    table = list(zip(x1[100:], x2[100:], y[100:], log_y))
    print(tabulate(table, headers=["x1", "x2", "y", "new y"], tablefmt="github"))
    print(loss)
