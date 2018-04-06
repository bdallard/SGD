#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Stochatic gradiant descent (SGD) Algorithm 
		
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs
import argparse

#les fonctions d'activation 
def sigmoid_function1(x): 
	return 1.0/(1+np.exp(-x))

def sigmoid_function2(x):
	return np.tanh(x)
	
#fonctions pour changer d'ensemble d'entrainement 
def next_batch(X, y, batchsize):
	for i in np.arange(0, X.shape[0], batchsize):
		#retourne un object generator 
		yield (X[i: i+batchsize], y[i: i+batchsize])
	
#affichage de l'etat entrainement via parsing 
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of SGD mini-batches")
args = vars(ap.parse_args())

#generation de data random --> Sklearn.datasets.make_blobs 
(X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
print("*** Shape des vecteurs ***\n")
print("X shape : ", X.shape)  #(400 points, 2classes) 
print("Y shape : " , y.shape) 

print("\n [INFO] début de l'entrainement ...")



#ALGO 
	
X = np.c_[np.ones((X.shape[0])), X] #np.c_ := concatenation 2 matrices  
W = np.random.uniform(size=(X.shape[1],))
#tableau pour stocker la perte 
lossHistory=[]
print("*** Poids avant la backpropagation ***\n")
#print(W)

#boucle sur les ensemble d'entrainement afin de mettre a jour le reseau
for epoch in np.arange(0,args["epochs"]): 
	epochLoss=[]
	#iteration sur les ensembles d'entrainement
	for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
		preds = sigmoid_function1(batchX.dot(W))
		#calcul de l'erreur
		error = preds - batchY
		loss = np.sum(error ** 2)
		epochLoss.append(loss)
		gradient = batchX.T.dot(error) / batchX.shape[0]
		#mise a jours des poids 
		W += args["alpha"]*gradient
	#incrementation du tab de perte	
	lossHistory.append(np.average(epochLoss))

print("*** Poids après la backpropagation ***\n")
print(W)

#solution Y 
Y = (-W[0] - (W[1]*X)) / W[2]
#plot de la solution 
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")

#loss au cours des itération
fig = plt.figure()
#plot de la perte
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

print(0)
