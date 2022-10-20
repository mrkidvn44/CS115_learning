import csv
import random

import numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def BMI_cal(w, h):
    return w/h**2


def data_create():
    data = []
    w = round(random.uniform(20, 120), 1)
    h = round(random.uniform(1, 1.9), 1)
    o = 0
    if BMI_cal(w, h) >= 30:
        o = 1
    data.append(str(w))
    data.append(str(h))
    data.append(str(o))
    return data


def sigmol(x):
    return round(1/(1+np.exp(-x)),4)


def predict(w, data):
    tmp = 0
    for i in range(len(data)):
        tmp += w[i] * data[i]
    tmp += w[2]
    return sigmol(tmp)


def grad_reg(w, y, p, data, lr):
    for i in range(len(data)):
        for j in range(len(w) - 1):
            w[j] += (y[i] - p[i])*data[i][j]*lr
        w[2] += (y[i] - p[i])*lr
    return w

fields = ['Weight', 'Height', 'Obesed']
rows = []
for i in range(1025):
    rows.append(data_create())


filename = "data_set.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)

path = "./data_set.csv"
file = pd.read_csv(path)
y = file.pop("Obesed")

X_train, X_val, Y_train, Y_val = train_test_split(file,y,train_size=0.9, random_state=0)
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
Y_train = Y_train.to_numpy()
Y_val = Y_val.to_numpy()

w = np.zeros(3)
lr = 0.002
step,max_step = 0, 30000
for step in range(max_step):
    p = []
    for i in range(len(X_train)):
        p.append(predict(w,X_train[i]))

    w = grad_reg(w, Y_train, p, X_train, lr)

test = []
print("Nhap can nang:")
test.append(float(input()))
print("Nhap chieu cao:")
test.append(float(input()))
print(predict(w,test))
