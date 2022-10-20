import math
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

def sigmoid(x):
    return round(1/(1 + np.exp(-x)), 6)


def loss_cal(y, p):
    Loss = 0
    for i in range(len(p)):
        Loss += y[i] * math.log(p[i]) + (1-y[i]) * math.log(1-p[i])
    return Loss


def predict(w, data):
    p = np.dot(w, data.T)
    p = sigmoid(p)
    return p


def min_max_scale(x):
    z = np.zeros(len(x))
    min_x, max_x = min(x), max(x)
    for i in range(len(x)):
        z[i] = float((x[i]-min_x)/(max_x - min_x))
    return z


def gradient_regression(w, p, y, lr, x):
    w += np.dot(y - p, x) * lr/len(y)
    return w


def evaluate(p, y):
    matrix = np.zeros((2, 2))
    for i in range(len(p)):
        matrix[round(y[i])][round(p[i])] += 1
    tn, tp, fn, fp = matrix[0][0], matrix[1][1], matrix[1][0], matrix[0][1]
    return round((tp + tn)/(tn+tp+fn+fp), 4), tp, tn, fp, fn



path = "./heart.csv"
file = pd.read_csv(path)
y = file.pop('target')
X_train, X_val, Y_train, Y_val = train_test_split(file, y, train_size=0.67, random_state=0)
name = X_train.columns
Y_train = Y_train.to_numpy()
Y_val = Y_val.to_numpy()
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
model_1= tree.DecisionTreeClassifier(max_depth=5, criterion='entropy')
model_1= model_1.fit(X_train, Y_train)
p_1= model_1.predict(X_val)
model= tree.DecisionTreeClassifier(max_depth=5, criterion='gini')
model= model.fit(X_train, Y_train)
p= model.predict(X_val)
name = name.to_numpy()
print(name)
print(evaluate(p, Y_val))
print(evaluate(p_1, Y_val))
plt.figure(figsize=(30, 30))
tree.plot_tree(model, feature_names=name, label=None)
plt.show()