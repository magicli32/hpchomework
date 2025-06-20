# -*- coding = utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

data = pd.read_csv("num_solution.txt", header=None)
y_num = data.values.reshape(101, )
x = np.arange(0, 1.01, 0.01)
y_exact = math.exp(-3.14**2)*np.sin(3.14*x)

# calculate the error
e = np.max(np.abs(y_exact - y_num))
print(e)

# plot figure
plt.figure(figsize=(8, 5))
plt.xlabel("x", fontsize=15)
plt.ylabel("Temperature", fontsize=15)
plt.plot(x, y_exact, color="b", linewidth=1.5, label="exact solution")
plt.plot(x, y_num, color="r", linewidth=1.5, label="numerical solution")
plt.legend()
plt.savefig(fname="./verification.png", dpi=300)
plt.show()
