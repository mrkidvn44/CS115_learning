import math

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def sigmol(x):
    return round(1/(1 + np.exp(-x)), 4)


def loss_cal(y, p):
    Loss = 0
    for i in range(len(p)):
        Loss += y[i] * math.log(p[i]) + (1-y[i]) * math.log(1-p[i])


def predict(w, data):
    temp = 0
    for i in range(len(w)-1):
        temp += w[i] * data[i]
    temp += w[13]
    p = sigmol(temp)
    return p


def min_max_scale(x):
    z = np.zeros(len(x))
    min_x, max_x = min(x), max(x)
    for i in range(len(x)):
        z[i] = float((x[i]-min_x)/(max_x - min_x))
    return z


def gradient_regression(w, p, y, lr, x):
    for i in range(len(p)):
        for j in range(len(w) - 1):
            w[j] += (y[i] - p[i])*x[i][j]*lr
        w[13] += (y[i] - p[i])*lr
    return w


def evaluate(p, y):
    tp, tn,fn, fp = 0, 0, 0, 0
    for i in range(len(y)):
        if round(p[i]) == y[i]:
            if y[i] == 1:
                tp +=1
            else:
                tn +=1
        else:
            if round(p[i]) == 0:
                fn +=1
            else:
                fp +=1
    Zeros = []
    Zeros.extend([0, 0, tn, fp])
    Ones = []
    Ones.extend([tp, fn, 0, 0])
    return round((tp + tn)/(tn+tp+fn+fp), 4), tp, tn, fp, fn



path = "./heart.csv"
file = pd.read_csv(path)
y = file.pop('target')

X_train, X_val, Y_train, Y_val = train_test_split(file, y, train_size=0.7, random_state=0)
Y_train = Y_train.to_numpy()
X_train = X_train.to_numpy()
X_train_ones = X_train[[index for index in [i for i in np.where(Y_train == 1)][0]]]
X_train_zeros = X_train[[index for index in [i for i in np.where(Y_train == 0)][0]]]
X_train = X_train_ones[np.random.randint(X_train_ones.shape[0], size=400)]
X_train = np.append(X_train, X_train_zeros[np.random.randint(X_train_zeros.shape[0], size=200)], axis = 0)
Y_train = np.ones(400)
Y_train = np.append(Y_train, np.zeros(200))
X_train = np.inser(X_train, np.zeros(600))
