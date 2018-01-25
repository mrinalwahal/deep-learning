# The Following Program is Designed for the XOR Function.
# Project Designed and Maintained by Mrinal Wahal.

import numpy as np

def sigmoid(x, derivative = False):
    if derivative == True: return x(x-1)
    return (1/(1 + np.exp(-x)))

X = np.array([ [0,1],[1,1],[1,0],[0,0] ])
y = np.array([[1,0,1,0]]).T

global syn0
syn0 = 2*np.random.random((2,4)) - 1
global syn1
syn1 = 2*np.random.random((4,1)) - 1

def think(input_layer, syn0, syn1):
    l1 = sigmoid(np.dot(input_layer,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    return l2

def train(input_layer, output_layer, syn0, syn1):
    for j in xrange(60000):
        l1 = sigmoid(np.dot(input_layer,syn0))
        l2 = sigmoid(np.dot(l1,syn1))
        l2_delta = (output_layer - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += l1.T.dot(l2_delta)
        syn0 += input_layer.T.dot(l1_delta)

print "-"*60
print "Before Training."
print think(X, syn0, syn1)
print "After Training."
train(X,y, syn0, syn1)
print think(X, syn0, syn1)
print "-"*60

print "New Inputs."
x1 = np.array([ [0,0], [1,0], [1,1], [0,1]])
print think(x1, syn0, syn1)