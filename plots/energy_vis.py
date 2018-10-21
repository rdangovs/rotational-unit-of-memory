import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

lin_coord = np.load("linear_coord.npy")
lin_height = np.load("linear_height.npy")

con_coord = np.load("contour_coord.npy")
con_height = np.load("contour_height.npy")

print(lin_coord)
print(lin_height)
print(con_coord)
print(con_height)

print(con_height.shape)
print(lin_coord.shape)

plt.plot(lin_coord, lin_height)
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(con_coord, con_coord, con_height)
ax.clabel(CS, inline=1, fontsize=10)

plt.show()
