import numpy as np
import theano
import theano.tensor as T



class layer(object):
	def __init__(self, insize, outsize, weights= None, bias =None):
		self.inputsize = insize
		self.outputsize = outsize
		if weights is None:
			W_bound= np.sqrt(self.inputsize * self.outputsize)
			w = np.random.uniform( low = -1. / W_bound, high= 1./ W_bound, size = (self.inputsize,self.outputsize))
			self.weights = theano.shared(name = 'weights', value = w.astype(theano.config.floatX), borrow = True)
		else:
			self.weights = theano.shared(name = 'weights', value = weights, borrow = True)
		if bias is None:
			b_vals = np.random.uniform(size=self.outputsize)
			self.bias = theano.shared(name='bias', value = b_vals.astype(theano.config.floatX))
		else:
			self.bias = theano.shared(name = 'bias', value = bias.astype(theano.config.floatX))

def testtrain(sample, expected, iteration, layer1 = None):	
	vectorinlength = sample.shape[1]
	vectoroutlength = expected.shape[1]
	datasize = sample.shape[0]
	if layer1 is None:
		layer1 = layer(vectorinlength, vectoroutlength)
	else :
		layer1 = layer1
	x = T.dmatrix('x')
	y = T.dmatrix('y')
	finaloutput = T.tanh(T.dot(x, layer1.weights) + layer1.bias )	
	err = ((finaloutput - y)**2).mean()
 	f = theano.function([x,y], err, updates =((layer1.weights ,layer1.weights - (0.01/datasize/datasize)* T.grad(err, layer1.weights)), (layer1.bias, layer1.bias - (0.01/datasize/datasize)* T.grad(err, layer1.bias))))
	totalerr = 0
	print(datasize)
	for i in range(iteration):
		totalerr = f(sample, expected)
		print(totalerr)

	return layer1

def train (sample, expected, iteration, layers= None):
	'''
	sample and expected are matrices where each rows are a single datum
	and the size of the rows are the size of the datum
	'''
	vectorinlength = sample.shape[1]
	vectoroutlength = expected.shape[1]
	datasize = sample.shape[0]
	if layers is None:
		layer1 = layer(vectorinlength, 30)
		layer2 = layer(30, 10)
		layer3 = layer(10, vectoroutlength)		
	else :
		layer1 = layers[0]
		layer2 = layers[1]
		layer3 = layers[2]
	x,y = T.dmatrices('x', 'y')
	firstoutput  = T.tanh(T.dot(x,           layer1.weights) + layer1.bias)
  	secondoutput = T.tanh(T.dot(firstoutput, layer2.weights) + layer2.bias)
	finaloutput  = T.tanh(T.dot(secondoutput,layer3.weights) + layer3.bias)	
	err = ((finaloutput - y)**2).mean()
 	f = theano.function([x,y], err, updates =((layer1.weights ,layer1.weights - 0.05* T.grad(err, layer1.weights)), (layer1.bias, layer1.bias - 0.05* T.grad(err, layer1.bias)), (layer2.weights,layer2.weights -0.05* T.grad(err, layer2.weights)), (layer2.bias, layer2.bias - 0.05* T.grad(err, layer2.bias)), (layer3.weights, layer3.weights -(0.05/datasize/datasize)* T.grad(err, layer3.weights)), (layer3.bias, layer3.bias - 0.05* T.grad(err, layer3.bias))))
	totalerr = 0
	for i in range(iteration):
		totalerr = f(sample, expected)
		print(totalerr)
	return layer1,layer2,layer3

def predict(data , layers):
	layer1 = layers[0]
	layer2 = layers[1]
	layer3 = layers[2]
	firstoutput = T.tanh(T.dot(x,           layer1.weights) + layer1.bias )
  	secondoutput =T.tanh(T.dot(firstoutput, layer2.weights) + layer2.bias )
	finaloutput = T.tanh(T.dot(secondoutput,layer3.weights) + layer3.bias	)
	return finaloutput

def examine(sample, expected, layers):
	layer1 = layers[0]
	layer2 = layers[1]
	layer3 = layers[2]
	samplesize = sample.shape[0]
	x,y = T.dmatrices('x', 'y')
	firstoutput = 1 / (1 + T.exp( -T.dot(x,           layer1.weights) - layer1.bias ))
  	secondoutput = 1/ (1 + T.exp(- T.dot(firstoutput, layer2.weights) - layer2.bias))
	finaloutput = T.dot(secondoutput, layer3.weights) + layer3.bias	
	err = ((finaloutput - y)**2).sum()
	f = theano.function([x,y], err)
	outcome = f(sample, expected)
	result = outcome/samplesize
	return result

totalsample = np.loadtxt("/home/abel/Desktop/pricelist.txt", delimiter = " ")

situation = totalsample[:,0:30]

result = totalsample[:,30:35]

