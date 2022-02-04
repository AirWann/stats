from distutils.log import error
import numpy as np
import matplotlib.pyplot as  plt
""" import pandas as pd """
import numpy.random as rdm

##Méthode de rejet
#Implémentation de la densité
def f(x):
    if x<1:
        return (2/3) * x
    else:
        return (2/3)

c = 4/3
cg = 2/3

#Implémentation de la méthode de rejet
def tiragedef (f) :
    Y = rdm.uniform(0,2)
    U = rdm.uniform(0,1)
    while U > (f(Y)/cg):
        Y = rdm.uniform(0,2)
        U = rdm.uniform(0,1)
    return Y

##Représentation de la variable aléatoire obtenue
nbtirages = 10**6
cases = 100
T = np.linspace(0,2,cases)
NBvals = np.zeros(cases)
for k in range(nbtirages):
    x = tiragedef(f)
    case = int(x/2*cases)
    NBvals[case] = NBvals[case] + 1

NBvals = (NBvals * cases) /2 / nbtirages
Ttheo = np.vectorize(f)(T)
plt.ion()
plt.figure()
plt.clf()
plt.plot(T,NBvals)
plt.plot(T,Ttheo)

plt.show()

##Approximation de l'intégrale par la méthode de Monte Carlo
u = lambda x : np.exp(x**2)

def monteCarloRejet(u,f,n):
    X = np.zeros(n)
    sum = 0

    for k in range(n):
        X[k] = u(tiragedef(f))

    In = np.sum(X)/n
    Sn2 = np.sum((X-In)**2)/(n-1)

    return(In,Sn2)

N = [10**n for n in range(2,7)]
approx = np.zeros_like(N)
inter = np.zeros_like(N)
for k,n in enumerate(N) :
    approx[k], inter[k] = monteCarloRejet(u,f,n)
    inter[k] = inter[k]/n**0.5

plt.figure()
plt.clf()
plt.semilogx(N/np.log(10),approx, label = "Intégrale approchée")
plt.semilogx(N/np.log(10),inter, label = "Largeur de l'intervalle de confiance")
plt.legend()

plt.title("Approximation de l'intégrale par la méthode de Monte Carlo")

m=monteCarloRejet(u,f,N[-1])
a,b = m[0] - m[1]/N[-1]**0.5, m[0] + m[1]/N[-1]**0.5
print("L'intégrale considérée vaut", m[0], "avec l'intervalle de confiance associé [", a, ",", b, "]")

###Exercice 11
##Estimation de J
#Simulation de la loi Pareto(1)
theta = 1
g=lambda t: (1-t)**(-1/theta)
u=lambda t: t**2/(1+t**3)

def monteCarloInversion(u,g,n):
    n = int(n)
    fX = np.zeros(n)
    sum = 0

    for k in range(n):
        fX[k] = u(g(rdm.uniform(0,1)))

    In = np.sum(fX)/n
    Sn2 = np.sum((fX-In)**2)/(n-1)

    return (In,Sn2)

##Approximation de J par la méthode de Monte Carlo
N = np.zeros(8-2)
approx1 = np.zeros_like(N)
var1 = np.zeros_like(N)
for k in range(8-2):
    N[k] = 10**(k+2)
    approx1[k],var1[k] = monteCarloInversion(u,g,int(N[k]))

plt.figure()
plt.clf()
plt.semilogx(N/np.log(10),approx1)
plt.title("Approximation de J, première méthode")

m=monteCarloInversion(u,g,N[-1])
a,b = m[0] - m[1]/N[-1]**0.5, m[0] + m[1]/N[-1]**0.5
print("Avec la première méthode, J =", m[0], "avec l'intervalle de confiance associé [", a, ",", b, "]")

##Approximation de J par la méthode de Monte Carlo sur une autre expression
g=lambda t: t
u=lambda t: t/(1+t**3)

N = np.zeros(8-2)
approx2 = np.zeros_like(N)
var2 = np.zeros_like(N)
for k in range(8-2):
    N[k] = 10**(k+2)
    approx2[k],var2[k] = monteCarloInversion(u,g,int(N[k]))

plt.figure()
plt.clf()
plt.semilogx(N/np.log(10),approx2)

plt.title("Approximation de J, seconde méthode")

m=monteCarloInversion(u,g,N[-1])
a,b = m[0] - m[1]/N[-1]**0.5, m[0] + m[1]/N[-1]**0.5
print("Avec la seconde méthode, J =", m[0], "avec l'intervalle de confiance associé [", a, ",", b, "]")

##Comparaison entre les deux méthodes
plt.figure()
plt.clf()
plt.semilogx(N/np.log(10),np.abs(approx2-approx1))

plt.title("Différences entre les deux approximations")

plt.figure()
plt.clf()
plt.semilogx(N/np.log(10),np.abs(var2-var1))

plt.title("Différences entre les deux variances")
