import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class BinaryClassifier:
	def __init__(self, input_dim, hidden_layers, weights_initialization="uniform", activation="tanh"):
		# Activation Functions and their derivatives
		self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
		self.relu = lambda x: np.where(x > 0, x, 0)
		self.sigmoid_der = lambda x: x * (1 - x)
		self.relu_der = lambda x: np.where(x > 0, 1, 0)
		self.tanh = lambda x: (2 / (1 + np.exp(-2 * x))) - 1
		self.tanh_der = lambda x: 1 - x ** 2
		self.activation = activation
		self.input_dim = input_dim
		self.output_dim = 1
		layers = []
		layers.append(input_dim)
		for i in hidden_layers:
			layers.append(i)
		layers.append(1)
		
		self.network = []
		self.outputs = []
		self.bias = [np.zeros(l) for l in layers[1:]]
		self.delta = []
		# Building network : Each layer = [W^[l-no], W^[l_no + 1]]
		for l_no in range(len(layers) - 1):
			if weights_initialization == "uniform":
				# Initialising weights as random
				self.network.append(np.random.rand(layers[l_no], layers[l_no + 1]))
			elif weights_initialization == "normal":
				self.network.append(np.random.normal(size=(layers[l_no], layers[l_no + 1])))
			elif weights_initialization == "ones":
				self.network.append(np.full((layers[l_no], layers[l_no + 1]), 1.0))
	
	# Trains the network and returns a list of accuracies and loss over the training epochs
	def train(self, X, Y, lr=0.5, epochs=10, use_bias=False):
		acc_list = []
		loss_list = []
		fscore_list = []
		for epoch in range(epochs):
			self.forward_pass(X)
			self.backward_pass(Y, use_bias)
			self.update_weights(lr)
			acc, loss, fscore = self.evaluate(X, Y)
			acc_list.append(acc)
			loss_list.append(loss)
			fscore_list.append(fscore)
		return [acc_list, loss_list, fscore_list]
	
	# Does forward pass and stores the output for all of the training examples in outputs
	def forward_pass(self, X):
		self.outputs = []
		# First Layer outputs are the inputs itself
		self.outputs.append(X)
		for l_no, layer in enumerate(self.network):
			layer = np.append(layer, self.bias[l_no].reshape(1, self.bias[l_no].shape[0]), axis=0)
			inputs = np.append(self.outputs[-1], np.full((self.outputs[-1].shape[0], 1), 1), axis=1)
			if l_no != len(self.network) - 1:
				if self.activation == "relu":
					# ReLU is activation for hidden layers
					self.outputs.append(self.relu(np.matmul(inputs, layer)))
				elif self.activation == "tanh":
					self.outputs.append(self.tanh(np.matmul(inputs, layer)))
			else:
				# Sigmoid is activation for last layer
				self.outputs.append(self.sigmoid(np.matmul(inputs, layer)))
	
	# L2 loss is assumed
	def backward_pass(self, Y, use_bias):
		deltas = []
		last = True
		dels = []
		# Last Layer, dZ/dx = out - y for each example
		for i in range(Y.shape[0]):
			out = self.outputs[-1][i] - Y[i]
			dels.append(out)  # np.multiply(out,self.sigmoid_der(self.outputs[-1][i])))
		deltas.append(np.array(dels).reshape(Y.shape[0], 1))
		
		# for other layers dZ/dx = sum(Weight*dZ/dx of next layer)*(activation_der(outputs of the layer))
		for l_no, layer in enumerate(reversed(self.network)):
			if l_no == len(self.network) - 1:
				break
			dels = []
			for j in range(Y.shape[0]):
				out = []
				for i in range(layer.shape[0]):
					out.append(np.dot(layer[i], deltas[-1][j]))
				if self.activation == "tanh":
					dels.append(np.multiply(out, self.tanh_der(self.tanh(self.outputs[-(l_no + 2)][j]))))
				elif self.activation == "relu":
					dels.append(np.multiply(out, self.relu_der(self.outputs[-(l_no + 2)][j])))
			deltas.append(np.array(dels).reshape(Y.shape[0], len(layer)))
		self.deltas = [d for d in reversed(deltas)]
		if use_bias:
			self.bias = [(1 / len(Y)) * np.sum(np.array(d), axis=0) for d in self.deltas]
	
	# Updates weights using the deltas calculated in the backward pass
	def update_weights(self, lr):
		num = len(self.outputs[0])
		for l_no in range(len(self.network)):
			for j in range(num):
				# dW = (1/num)*(out*del)
				# W = W - lr*dW (SGD)
				self.network[l_no] -= lr * (1 / num) * np.matmul(
					self.outputs[l_no][j].reshape(len(self.outputs[l_no][j]), 1),
					self.deltas[l_no][j].reshape(1, len(self.deltas[l_no][j])))
	
	def evaluate(self, X, Y):
		tp = 0  # True positives
		fp = 0  # False positives
		tn = 0  # True negatives
		fn = 0  # False negatives
		self.forward_pass(X)
		for n in range(X.shape[0]):
			if self.outputs[-1][n] >= 0.5:
				y_pred = 1
			else:
				y_pred = 0
			# print(Y[n], y_pred)
			if Y[n] == y_pred:
				if Y[n] == 1:
					tp += 1
				else:
					tn += 1
			else:
				if y_pred == 1:
					fp += 1
				else:
					fn += 1
		loss = 0
		# loss = half*sum(out - y)^2
		for n in range(X.shape[0]):
			loss += (self.outputs[-1][n] - Y[n]) ** 2
		
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		fscore = (2 * precision * recall) / (precision + recall)  # Harmonic mean of precision and recall
		
		return ((tp + tn) / X.shape[0]) * 100, loss / len(X), fscore

 
input_dim = 10
hidden_layers = [20, 20]
weights_initialization = "normal"
activation = "tanh"
nn = BinaryClassifier(input_dim, hidden_layers, weights_initialization, activation)
print("Input dimension = ", input_dim)
print("Hidden Layers = ", hidden_layers)
print("Weights Initialising = ", weights_initialization)
print("Activation Function = ", activation)

df = pd.read_csv('housepricedata_a2_2.csv')
label = 'AboveMedianPrice'
X = df.drop(label, axis=1).values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

Y = df[label].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
lr = 0.1
epochs = 1000
print("Training: (lr = ", lr, " epochs = ", epochs, ")")
metrics = nn.train(X_train, Y_train, lr, epochs)
# print(nn.bias)

plt.plot(np.arange(len(metrics[0])), metrics[0])
plt.savefig("Training Accuracy.eps")
plt.close()

plt.plot(np.arange(len(metrics[1])), metrics[1])
plt.savefig("Training Loss.eps")
plt.close()

plt.plot(np.arange(len(metrics[2])), metrics[2])
plt.savefig("Training Fscore.eps")
plt.close()

print("Evaluation:")
print(nn.evaluate(X_test, Y_test))
