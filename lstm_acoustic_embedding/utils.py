import numpy

def sigmoid(x):
    return 1./(1. + numpy.exp(numpy.clip(-x, -30, 30)))

def tanh(x):
    return 2./(1. + numpy.exp(numpy.clip(-2*x, -30, 30))) - 1.0
