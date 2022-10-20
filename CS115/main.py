import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


def loss_cal(W, B, data):
  Loss = 0
  for k in range(10):
    Loss += ((data[k][1] - W*data[k][0] - B)**2)
  return Loss/10


if __name__ == '__main__':
    w, b = 0, 0
    Data = [[2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 5.5], [9, 8], [10, 7.5], [12, 9.5], [12.5, 9]]
    f = open("log.txt", "w")
    learning_rate = 0.0001
    loss = []
    Time = []
    log_w = []
    log_b = []
    for j in range(100000):
        dw_loss = 0
        db_loss = 0
        for i in range(10):
            dw_loss += -2*Data[i][0]*(Data[i][1] - w*Data[i][0] - b)
            db_loss += -2*(Data[i][1] - w*Data[i][0] - b)
        dw_loss = dw_loss/10
        db_loss = db_loss/10
        w = w - learning_rate*dw_loss
        b = b - learning_rate*db_loss
        if j < 500:
            loss.append(loss_cal(w, b, Data))
            log_w.append(w)
            Time.append(j)
        f.write(str(loss_cal(w, b, Data)) + "\n")

    loss = np.array(loss)
    Time = np.array(Time)
    log_w = np.array(log_w)
    new_Time = np.linspace(Time.min(), Time.max(), 500000)
    spline = make_interp_spline(Time, loss, k=3)
    new_Loss = spline(new_Time)
    plt.subplot(3, 1, 1)
    plt.plot(new_Time, new_Loss, 'or')
    plt.subplot(3, 1, 2)
    for i in range(10):
        plt.plot(Data[i][0], Data[i][1], 'ob')
    plt.plot(np.array([0, 14]), np.array([w*0 + b, w*14 + b]))
    #new_loss = np.linspace(loss.min(), loss.max(), 500000)
    #spline = make_interp_spline(loss, log_w, k=3)
    #new_w = spline(new_loss)
    plt.subplot(3, 1, 3)
    for i in range(len(log_w)):
        plt.plot(log_w[i], loss[i], 'or')
    plt.show()
    f.close()
    print(w, b)

