import numpy as np
import math as ma


def graddescent(A,x1,b,ep):
    x = x1
    i = 0
    learnrate = 0.1
    grad = 2*(A*x+b)
    while (np.linalg.norm(grad) > ep):
        i = i + 1
        x = x - learnrate*grad
        grad = 2*(A*x+b)
        newfunc = x.T*A*x + 2*b*x
        print("Numbe of Iter:",i)
        print("Grad:",grad)
        print("New value:",newfunc)
    return newfunc


A = np.array([1,1])
x1 = np.array([2,1])
b = np.array([0,0])
ep = ma.exp(-10)



graddescent(A,x1,b,ep)






