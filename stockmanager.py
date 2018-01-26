import theano
import theano.tensor as T
import numpy as np
import math

x = T.dmatrix('x')
y = T.dmatrix('y')
w = T.dmatrix('w')
b = T.dvector('b')

inputvectors = T.stack(price, memory, status)
Aw1 = theano.shared()

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
	def sigmoid(self, inputvector):
		res = T.nnet.sigmoig(T.dot(inputvector, self.weights) + self.bias)
		return res

def neuralprocess(layers, inputvector):
	outcome = inputvector	
	for layer in layers:
		outcome = layer.sigmoid(outcome)
	return outcome	

def decisionmakers(inputsize, outputsize):
	layer1 = layer(inputsize, 30)
	layer2 = layer(30, 10)
	layer3 = layer(10, outputsize)
	return layer1, layer2, layer3

def memorizer(inputsize, memorysize)
	layer1 = layer(inputsize, 30)
	layer2 = layer(30, 10)
	layer3 = layer(10, outputsize)
	return layer1, layer2, layer3

def evaluation(price, status)
	result = T.dot( price, status)	
	return result

def executor(decision, price, status):
	for i in range(len(decision)):
		amount = math.floor(decision[i])
		status[0] = status[0] - amount * price[i +1] - 0.01* abs(amount) * price(i + 1) 
		status[i+1] + amount
	return status
				

#status = [current money, # of stock]
# price = [1, price of company i stock]
#decision = [ quantity of stock]
def decisionmaker(price, memory = None, status, duration = 5, decisionlayer = None, memorylayer = None):
	price.shape[0] = duration	
	if memory is None:
		memory = np.zeros(500)
	if decisionlayer is None:
		dl1 = layer(memory.shape[0] + price.shape[0] + status.shape[0] , 50)
		dl2 = layer(50 , 10)
		dl3 = layer(10, 1)
		decisionlayer = [dl1, dl2, dl3]
	if memorylayer is None:
		ml1 = layer(memory.shape[0] + price.shape[0] + status.shape[0] , 500)
		ml2 = layer(500 , 500)
		memorylayer = [ml1, ml2]
	inputvector = T.concatenate(price, memoery, status)
	decision = neuralprocess(decisionlayer, inputvector) 
	new_memory = neuralprocess(memorylayer, inputvector)
	return decision, new_memory
		
def train(price, memory = None, status, duration = 5, decisionlayer = None, memorylayer = None):
	inputvector = T.concatenate(price, memoery, status)
	if memory is None:
		memory = np.zeros(500)
	if decisionlayer is None:
		dl1 = layer(memory.shape[0] + price.shape[0] + status.shape[0] , 50)
		dl2 = layer(50 , 10)
		dl3 = layer(10, 1)
		decisionlayer = [dl1, dl2, dl3]
	if memorylayer is None:
		ml1 = layer(memory.shape[0] + price.shape[0] + status.shape[0] , 500)
		ml2 = layer(500 , 500)
		memorylayer = [ml1, ml2]
	decision = T.nnet.softmax(neuralprocess(decisionlayer, inputvector))
	newstatus = status + decision
	totalasset = T.dot( price, status )
	cost = T.exp(-totalasset / 100000)
	
