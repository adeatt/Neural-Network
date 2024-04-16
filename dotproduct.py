import numpy as np                  #dot product of 4 input neurons and 3 output

inputs = [1, 2, 3, 2.5]

wheights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

output = np.dot(wheights, inputs  ) + bias
print(output)   