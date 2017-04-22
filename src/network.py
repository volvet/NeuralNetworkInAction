
import numpy as np


def sigmoid(z):
	return 1.0/(1.0+exp(-z))


class Network(object):

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


if __name__ == '__main__':
	print 'test network'
	sizes = [2, 3, 1]
	net = Network(sizes)
	print "biases: ", net.biases
	print "weights: ", net.weights