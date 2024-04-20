import numpy as np       
import nnfs 
from nnfs.datasets import spiral_data



nnfs.init()

X = [[1, 2, 3, 2.5],                                                    #multiple batch inputs from 4 input neurons
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(100, 3)                                              # dataset


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):                            #attributs n_inputs(how many input neurons) and n_neurons(how many neurons the layer has).
        self.wheights = 0.10 * np.random.randn(n_inputs, n_neurons)     #wheigts are being randomly generatet for the amount of n_inputs and n_neurons (this example for 4 input and 5 output ) for every neuron (there are 5 neurons) in layer 1 we have 4 wheigts from the neurons in the input
        self.biases = np.zeros((1, n_neurons))                          # randomise biases. the amount of neurons in layer 1/2 for n_neurons are assinged a number (exp: layer 1 gets 5 biases for 5 neurons)
    def forward(self, inputs):
        self.output = np.dot(inputs, self.wheights) + self.biases       #output
    

class Activision_ReLU:                                                  #activation function
    def foward(self, inputs):
        self.output = np.maximum(0, inputs) 




layer1 = Layer_Dense(2,5)                                           
activation1 = Activision_ReLU()

layer1.forward(X)   


activation1.foward(layer1. output)
print(activation1.output)