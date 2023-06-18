#!/usr/bin/env python3
"""
Created on Tue Apr 19 19:07:36 2022
@author: dishant

"""

import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import math

#function for tsp48 where X is not random between (0,1)

def tsp(n,A):

    N = n
    X = A
    X=X/10000
    M=5*N
    kappa = 20
    c = np.zeros([M])
    c[0] = 2
    c[1] = -1
    c[M-1] = -1
    L = la.circulant(c)
    Lreg = kappa * L + np.eye(M)
    Lreg_inv = la.inv(Lreg)
    mu = np.mean(X, axis=0)
    eps = 0.001
    #Initializing Y matrix
    Y = eps*np.random.randn(M,2)+mu
    # Y2=Y

    #Calculating L2 norm
    x1 = np.outer(X[:,0], np.ones(M))

    y1 = np.outer(np.ones(N), Y[:,0])

    x2 = np.outer(X[:,1], np.ones(M))

    y2 = np.outer(np.ones(N), Y[:,1]) 
    DistXY = (x1-y1)**2 + (x2-y2)**2 
    beta = 10

    #Initializing P matrix

    softmax_XY = np.exp(-beta*DistXY)

    softmax_XY = softmax_XY / np.outer(np.sum(softmax_XY, axis=1), np.ones(M))

    iterations = 2000

    gamma = 1.05

    tour_length=0

    for k in range(0,iterations):
    #calculating l2 norm to update P and Y matrix 
        nDx1 = np.outer(X[:,0], np.ones(M))
        nDy1 = np.outer(np.ones(N), Y[:,0]) 
        nDx2 = np.outer(X[:,1], np.ones(M)) 
        nDy2 = np.outer(np.ones(N), Y[:,1]) 
        nDistXY = (nDx1-nDy1)**2 + (nDx2-nDy2)**2 
        softmax_XY_2 = np.exp(-beta*nDistXY)


    n_softmax = softmax_XY*softmax_XY_2 #Updating P matrix

    softmax_XY_2 = n_softmax / np.outer(np.sum(n_softmax,axis=1), np.ones(M))

    # Rate obtained by reducing kappa 
    kappa = kappa/gamma

    #Updating D matrix

    d = np.sum(softmax_XY_2.T,axis=1) 
    D = np.diag(d)

    Lreg = (kappa * L) + D 
    Lreg_inv=la.inv(Lreg) #Updating Y matrix

    Y = Lreg_inv.dot(softmax_XY_2.T.dot(X)) 
    softmax_XY = softmax_XY_2 #Calculating tour length

    for i in range(1,M): 
        y1=Y[i][1] 
        y2=Y[i-1][1] 
        x1=Y[i][0] 
        x2=Y[i-1][0] 
        dist=math.sqrt((y1-y2)**2+(x2-x1)**2) 
        tour_length+=dist


    print("Tour Length:", int(tour_length)) #added for post processing of the graph Y0=Y[0]

    Y = np.r_[Y,[Y0]]

    #plotting the graph of final sequence plt.scatter(X[:,0],X[:,1], color='red') plt.plot(Y[:,0],Y[:,1], color='black') plt.show()

    return softmax_XY


XM = np.empty((0,2), float)
#Reading x,y coordinates of TSP48 from txt file 
with open('tsp48.txt') as f:
    for line in f:
        x, y = line.split()
        X0 = [int(x),int(y)]
        print(X0) 
        XM=np.r_[XM,[X0]]
        print(x,y)

N=48
M=3*N
s=set()
sequence =[]

P=tsp(N,XM)

#Get the sequence and print it 
maxInC=P.argmax(axis=0)

for i in range(0,M):
    if maxInC[i] not in s: 
        s.add(maxInC[i]) 
        sequence.append(maxInC[i])

sequence.append(sequence[0]) 
print(sequence)

