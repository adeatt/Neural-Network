import numpy as np         

inputs = [[1, 2, 3, 2.5],                       #multiple batch inputs from 4 input neurons
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

wheights = [[0.2, 0.8, -0.5, 1.0],              #wheigts for neuron 1 on layer 1
            [0.5, -0.91, 0.26, -0.5],           #wheigts for neuron 2 on layer 1
            [-0.26, -0.27, 0.17, 0.87]]         #wheigts for neuron 3 on layer 1

bias = [2, 3, 0.5]                              # bias neuron 1,2,3 on layer 1

wheights2 = [[0.1, -0.13, -0.5],                #wheigts for neuron 1 on layer 2
            [0.5, 0.12, -0.33],                 #wheigts for neuron 2 on layer 2
            [-0.44, 0.73, -0.13]]               #wheigts for neuron 3 on layer 2

bias2 = [-1, 2, -0.5]                             #bias neuron 1,2,3 on layer 2


layer1_outputs = np.dot(inputs, np.array(wheights).T) + bias # outputs of layer 1 become the inputs of layer 2

layer2_outputs = np.dot(layer1_outputs, np.array(wheights2).T) + bias2 # outputs of layer 1 become the inputs of layer 2

print(layer2_outputs)