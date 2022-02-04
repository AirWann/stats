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

    

T = np.linspace(0,2,100)
NBvals = np.zeros(100)
for k in range(100):
    x = tiragedef(f)
    case = int(x/0.02)
    NBvals[case] = NBvals[case] + 1

plt.ion()
plt.figure()
plt.clf()
plt.plot(T,NBvals)

plt.show()



