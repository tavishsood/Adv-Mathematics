import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv", encoding="cp1251")

id_val = 102303246

alpha = 0.05 * (id_val % 7)
beta = 0.3 * ((id_val % 5) + 1)

raw_vals = data["no2"].dropna().values

final_vals = raw_vals + alpha * np.sin(beta * raw_vals)

print("alpha:", alpha)
print("beta:", beta)
print(final_vals[:5])

avg = np.mean(final_vals)
variance = np.var(final_vals)

inv_scale = 1 / (2 * variance)
norm_factor = 1 / np.sqrt(2 * np.pi * variance)

print("mean:", avg)
print("lambda:", inv_scale)
print("factor:", norm_factor)

grid = np.linspace(np.min(final_vals), np.max(final_vals), 500)
pdf_curve = norm_factor * np.exp(-inv_scale * (grid - avg) ** 2)

plt.hist(final_vals, bins=50, alpha=0.6)
plt.plot(grid, pdf_curve, "r")
plt.savefig("histogram.png")
plt.show()