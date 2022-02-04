from distutils.log import error
import numpy as np
import matplotlib.pyplot as  plt
""" import pandas as pd """
import numpy.random as rdm

def f(x):
    if x<0:
        print ("bzkehfgqkhdgsfkqjdgshf")
        return -1
    elif x>2:
        print ("baaaaaaaah (moi qui pleure)")
        return -1
    elif x<1:
        return (2/3) * x
    else:
        return (2/3)

c = 4/3
cg = 2/3

def tiragedef (f) :
    Y = rdm.uniform(0,2)
    U = rdm.uniform(0,1)
    while U > (f(Y)/cg):
        Y = rdm.uniform(0,2)
        U = rdm.uniform(0,1)
    return Y


nbtirages = 10**6
cases = 100
T = np.linspace(0,2,cases)
NBvals = np.zeros(cases)
for k in range(nbtirages):
    x = tiragedef(f)
    case = int(x/2*cases)
    NBvals[case] = NBvals[case] + 1


NBvals = (NBvals * cases) /2 / nbtirages
plt.ion()
plt.figure()
plt.clf()
plt.plot(T,NBvals)

plt.show()
