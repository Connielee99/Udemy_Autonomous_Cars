# this file implements some commonly used activation functions 

import numpy as np

def sigmoid(x):

	return 1 / ( 1 + np.exp(-x))

def tanh(x):

	numerator = 1 - np.exp(-2*x)
	denominator = 1 + np.exp(-2*x)

def ReLU(x):

	if x < 0:
		return 0
	return x

def leakyRelu(x, alpha = 0.01):

	if x < 0:
		return alpha * x
	return x

# exponential linear unit function
def ELU(x, alpha = 0.01):

	if x < 0:
		return alpha * (np.exp(x) - 1)
	return x

def swish(x, beta):

	return x * 2 * sigmoid(beta * x)

def softmax(x):

	numerator = np.exp(x)
	denominator = np.exp(x).sum(axis = 0)

	return numerator / denominator