import numpy as np
import matplotlib.pyplot as plt

e = np.loadtxt("evol.dat")
plt.plot(e)
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.show()

w = np.loadtxt("weights.dat")
plt.plot(w)
plt.xlabel("Timesteps")
plt.ylabel("Weights")
plt.show()

b = np.loadtxt("biases.dat")
plt.plot(b)
plt.xlabel("Timesteps")
plt.ylabel("Biases")
plt.show()

n = np.loadtxt("neural.dat")
plt.plot(n)
plt.xlabel("Timesteps")
plt.ylabel("Weights")
plt.show()
