import math
import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
Y_train = Y_train.to_numpy()
Y_val = Y_val.to_numpy()
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
#chon data output 0 va 1
X_train_ones = X_train[[index for index in [i for i in np.where(Y_train == 1)][0]]]
X_train_zeros = X_train[[index for index in [i for i in np.where(Y_train == 0)][0]]]
zerossize, onessize = 100, 300 #random.randint(200, len(X_train_zeros)), random.randint(175, len(X_train_ones))
X_train = X_train_ones[np.random.choice(X_train_ones.shape[0], onessize, replace=False), :]
X_train = np.append(X_train, X_train_zeros[np.random.choice(X_train_zeros.shape[0], zerossize, replace=False), :], axis=0)

Y_train = np.ones(onessize)
Y_train = np.append(Y_train, np.zeros(zerossize))

np.savetxt("foo.csv", Y_train, delimiter=",")
X_train = np.insert(X_train, 13, np.ones(len(X_train)), axis=1)
X_val = np.insert(X_val, 13, np.ones(len(X_val)), axis=1)

# shuffle = np.random.permutation(len(X_train))
# X_train = X_train[shuffle]
# Y_train = Y_train[shuffle]

for i in range(len(X_train.T)-1):
    X_train.T[i] = min_max_scale(X_train.T[i])

lr = 0.003
step, max_step = 0, 5000
w = np.zeros(14)
for step in range(max_step):
    p = []
    for i in range(len(X_train)):
        p.append(predict(w, X_train[i]))

    w = gradient_regression(w, p, Y_train, lr, X_train)
    if step % 500 == 0:
        print(step)

p_val = np.zeros(len(Y_val))
for i in range(len(X_val.T)-1):
    X_val.T[i] = min_max_scale(X_val.T[i])
for i in range(len(Y_val)):
    p_val[i] = predict(w, X_val[i])
acc, tp, tn, fp, fn = evaluate(p_val, Y_val)

df = pd.DataFrame(
    {"Tp": [None, tp], "Tn": [tn, None], "Fp": [fp, None], "Fn": [None, fn]},
    index=["0", "1"],
)


# define the colors for each column
colors = {"Tn": "#0B132B", "Tp": "#5BC0BE", "Fn": "#3A506B", "Fp": "#1C2541"}

fig = plt.figure(figsize=(10, 6))
ax = plt.gca()

# width of bars
width = 1

# create empty lists for x tick positions and names
x_ticks, x_ticks_pos = [], []

# counter for helping with x tick positions
count = 0

# reset the index
# so that we can iterate through the numbers.
# this will help us to get the x tick positions
df = df.reset_index()
# go through each row of the dataframe
print(df)
for idx, row in df.iterrows():
    # this will be the first bar position for this row
    count += idx

    # this will be the start of the first bar for this row
    start_idx = count - width / 2
    # this will be the end of the last bar for this row
    end_idx = start_idx
    # for each column in the wanted columns,
    # if the row is not null,
    # add the bar to the plot
    # also update the end position of the bars for this row
    for column in df.drop(["index"], axis=1).columns:
        if row[column] ==row[column]:
            plt.bar(count, row[column], color=colors[column], width=width, label=column)
            count += 1
            end_idx += width
    # this checks if the row had any not NULL value in the desired columns
    # in other words, it checks if there was any bar for this row
    # if yes, add the center of all the row's bars and the row's name (A,B,C) to the respective lists
    if end_idx != start_idx:
        x_ticks_pos.append((end_idx + start_idx) / 2)
        x_ticks.append(row["index"])

# now set the x_ticks
plt.xticks(x_ticks_pos, x_ticks)

# also plot the legends
# and make sure to not display duplicate labels
# the below code is taken from:
# https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
