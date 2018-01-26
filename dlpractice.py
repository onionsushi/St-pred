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
			b_vals = np.random.uniform(size=self.inputsize)
			self.bias = theano.shared(name='bias', value = b_vals.astype(theano.config.floatX))
		else:
			self.bias = theano.shared(name = 'bias', value = bias.astype(theano.config.floatX)
	def nextlayer(self, inputdata):
		output = T.dot(inputdata, self.weights) + self.bias
		return output

def traintutorial (inputmat, outputmat, w, b):
	inp, out = T.dvectors('inp', 'out')
	err = ((out - T.dot(inp, w) - b)**2).sum()
	f = theano.function([inp, out], err, updates = ((w, w- 0.05*T.grad(err,w)), (b, b- 0.05*T.grad(err, b))))
	totalerr = 0
	for i in range(inputmat.shape[0]):
		totalerr = totalerr + f(inputmat[i], outputmat[i])
	print(totalerr)

def testtrain (sample, expected, iteration):	
	vectorinlength = sample.shape[1]
	vectoroutlength = expected.shape[1]
	layer1 = layer(vectorinlength, vectoroutlength)
	for j in range(iteration):
		for i in range(sample.shape[0]):
			final = layer1.nextlayer(sample[i])
			err = ((expected[i] - final)**2).sum()
			gr_w1 = theano.function([],T.grad(err, layer1.weights))
			gr_b1 = theano.function([],T.grad(err, layer1.bias))
			layer1.weights.set_value(layer1.weights.get_value() - 0.05 *gr_w1())
			layer1.bias.set_value(layer1.bias.get_value() - 0.05 *gr_b1())
	print err
	return layer1

def train (sample, expected, iteration):
'''
	sample and expected are matrices where each rows are a single datum
	and the size of the rows are the size of the datum
'''
	vectorinlength = sample.shape[1]
	vectoroutlength = expected.shape[1]
	layer1 = layer(vectorinlength, vectorinlength)
	layer2 = layer(vectorinlength, vectoroutlength)
'''
	x,y = T.dmatrices('x', 'y')
	secondinput = 1 / (1 + exp( -T.dot(x, layer1.weights) - layer1.bias ))
  	predict = T.dot(secondinput, layer2.weight) - layer2.bias)
	err =  err = ((predict - y)**2).sum()
 	f = theano.function([x,y], err, 
					updates =((layer1.weights ,layer1.weights - 0.05* T.grad(err, layer1.weights))
(layer1.bias, layer1.bias - 0.05* T.grad(err, layer1.bias))
(layer2.weights,layer2.weights -0.05* T.grad(err, layer2.weights))
(layer2.bias, layer2.bias - 0.05* T.grad(err, layer2.bias))
 						 layer1.weights
'''
layer1.weights
layer1.weights
layer1.weights 
	for j in range(iteration):
		for i in range(sample.shape[1]):
			final = layer2.nextlayer(T.tanh(layer1.nextlayer(sample[i])))
			'''
			secondinput = 1 / (1 + exp( -T.dot(sample[i], layer1.weights) - layer1.bias ))
			final =  T.dot(secondinput, layer2.weight) - layer2.bias) 
			err = ((expected[i] - final)**2).sum()
			f = theano.function([],
'''
			err = ((expected[i] - final)**2).sum()
			gr_w1 = theano.function([],T.grad(err, layer1.weights))
			gr_w2 = theano.function([],T.grad(err, layer2.weights))
			gr_b1 = theano.function([], T.grad(err, layer1.bias))
			gr_b2 = theano.function([], T.grad(err, layer2.bias))
			layer1.weights.set_value(layer1.weights.get_value() - 0.05 *gr_w1())
			layer2.weights.set_value(layer2.weights.get_value() - 0.05 *gr_w2())
			layer1.bias.set_value(layer1.bias.get_value() - 0.05 *gr_b1())
			layer2.bias.set_value(layer1.bias.get_value() - 0.05 *gr_b2())
	print err
	return layer1,layer2

'''	
def test (sample, layers)
	for _ in layers
'''
a = np.asarray([[1,1,1],[2,2,2],[3,3,3.]])
b= np.asarray([1,2,3.])
