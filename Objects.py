import numpy as np         
np.random.seed(0)

X = [[1, 2, 3, 2.5],                                                    #multiple batch inputs from 4 input neurons
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):                            #attributs n_inputs(how many input neurons) and n_neurons(how many neurons the layer has).
        self.wheights = 0.10 * np.random.randn(n_inputs, n_neurons)     #wheigts are being randomly generatet for the amount of n_inputs and n_neurons (this example for 4 input and 5 output ) for every neuron (there are 5 neurons) in layer 1 we have 4 wheigts from the neurons in the input
        self.biases = np.zeros((1, n_neurons))                          # randomise biases. the amount of neurons in layer 1/2 for n_neurons are assinged a number (exp: layer 1 gets 5 biases for 5 neurons)
    def forward(self, inputs):
        self.output = np.dot(inputs, self.wheights) + self.biases       #output
    

layer1 = Layer_Dense(4,5)                                               #input has 4 neurons, layer 1 has 5
layer2 = Layer_Dense(5,2)                                               #input has 5 neurons(bc layer 1 had 5 as an output) and the layer 2 has 2 neurons

layer1.forward(X)                                                       # for the amount of neurons(defined by Layer_Dense(4,5)), biases(defined by self.biases) and the wheigts(defined self.wheigts) there is a output calculated. As the STARTING input we will be taking "X"
#print(layer1.output)                                                   #[[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]     #output of layer 1 for 5 neurons on the first run with X[0]
                                                                        #[-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]      #output of layer 1 for 5 neurons on the second run with X[1]
                                                                        #[-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]]     #output of layer 1 for 5 neurons on the third run with X[2]    

layer2.forward(layer1.output)                                           #same as layer 1 exept the amount of neurons for input are now the output of layer 1(so 5) and the output are now 2 neurons defined by layer2. as said the STARTING input is now layer1.output (with )
print(layer2.output)                                                    #[[ 0.148296   -0.08397602] # first run output for 2 neurons on layer 2 with input layer 1 for 5 neurons with layer1.output[0]
                                                                        #[ 0.14100315 -0.01340469] # second run with layer1.output[1]
                                                                        #[ 0.20124979 -0.07290616]] #t thrid run with layer1.output[2]