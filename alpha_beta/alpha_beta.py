from tkinter import N
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

data = pd.read_csv("alpha.txt", header=None)
y_num = data.values.reshape(1887, )

y_1000, y_500, y_200 = y_num[0:1001], y_num[1001:1502], y_num[1502:1703]
y_100, y_50, y_20, y_10 = y_num[1703:1804], y_num[1804:1855], y_num[1855:1876], y_num[1876:1887]


x_1000 = np.arange(0, 1.001, 0.001)
x_500 = np.arange(0, 1.002, 0.002)
x_200 = np.arange(0, 1.005, 0.005)
x_100 = np.arange(0, 1.01, 0.01)
x_50 = np.arange(0, 1.02, 0.02)
x_20 = np.arange(0, 1.05, 0.05)
x_10 = np.arange(0, 1.1, 0.1)
es = []

for x, y in zip([x_1000,x_500, x_200, x_100, x_50, x_20, x_10], \
    [y_1000, y_500, y_200, y_100, y_50, y_20, y_10]):
    y_exact = math.exp(-3.14**2)*np.sin(3.14*x)

    # calculate the error
    e = np.max(np.abs(y_exact - y))
    es.append(e)

# print(es)

# plot figure
dx = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
dx_log = np.log(dx)
es_log = np.log(es)
plt.figure(figsize=(8, 5))
plt.xlabel("log(dx)", fontsize=15)
plt.ylabel("log(e)", fontsize=15)
plt.plot(dx_log, es_log, color="b", linewidth=1.5, marker="*")
# plt.legend()
plt.savefig(fname="./alpha.png", dpi=300)
plt.show()
