from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

startTime = datetime.now()
print("\nStarted script at: {}".format(startTime))
print("\n* * * * BEGIN CODE EXECUTION * * * *\n")

class XORDeepNeuralNet:
  def __init__(self, 
               inputs,
               outputs,
               num_of_hidden_layers=2,
               neurons_per_hidden_layer=4,
               learning_rate=1.5,
               is_debugging=True):
    super(XORDeepNeuralNet, self).__init__()

    self.expected_output = outputs
    self.is_debugging = is_debugging
    self.learning_rate = learning_rate

    # Initialize our empty layers
    self.input_layer = inputs
    self.hidden_layers = []
    for _ in range(num_of_hidden_layers):
        self.hidden_layers.append(np.zeros((inputs.shape[0], neurons_per_hidden_layer)))
    self.output_layer = np.zeros((inputs.shape[0], outputs.shape[1]))

    # Initialize our random weight matricies
    self.hidden_weights = []

    # 1a. Input Layer -> First Hidden Layer
    self.hidden_weights.append(np.random.randn(self.input_layer.shape[1], neurons_per_hidden_layer))

    # 1b. Hidden Layers (assumption: num_of_hidden_layers >= 1)
    for layer in range(1, num_of_hidden_layers):
        self.hidden_weights.append(np.random.randn(self.hidden_weights[layer - 1].shape[1], neurons_per_hidden_layer))

    # 1c. Last Hidden Layer -> Output Layer
    self.output_weights = np.random.randn(neurons_per_hidden_layer, self.output_layer.shape[1])

    if (is_debugging):
        print("`inputs` {}, shape {}".format(inputs.tolist(), inputs.shape))
        print("`outputs` {}, shape {}".format(outputs.tolist(), outputs.shape))
        print("`num_of_hidden_layers` {}".format(num_of_hidden_layers))
        print("`neurons_per_hidden_layer` {}".format(neurons_per_hidden_layer))
        print("`learning_rate` {}".format(learning_rate))
        print("`self.input_layer` shape {}".format(self.input_layer.shape))
        for layer in range(num_of_hidden_layers):
          print("`self.hidden_layer_{}` shape {}".format(layer, self.hidden_layers[layer].shape))
        print("`self.output_layer` shape {}".format(self.output_layer.shape))
        for layer in range(num_of_hidden_layers):
          print("`self.hidden_weights_{}` shape {}".format(layer, self.hidden_weights[layer].shape))
        print("`self.output_weights` shape {}\n".format(self.output_weights.shape))

  def __call__(self, *args: Any, **kwds: Any) -> Any:
    # 1. Feed foward
    # 1a. Input Layer -> First Hidden Layer
    self.hidden_layers[0] = self.sigmoid(np.dot(self.input_layer, self.hidden_weights[0]))

    # 1b. Hidden Layers (assumption: num_of_hidden_layers >= 1)
    hidden_layer_len = self.hidden_layers.__len__()
    for layer in range(1, hidden_layer_len): # 1..n
        self.hidden_layers[layer] = self.sigmoid(np.dot(self.hidden_layers[layer - 1], self.hidden_weights[layer]))

    # 1c. Last Hidden Layer -> Output Layer
    self.output_layer = self.sigmoid(np.dot(self.hidden_layers[hidden_layer_len - 1], self.output_weights))

    # 2. Backpropagation
    # 2a. Get cost deriv. of weights: (last hidden layer) <-- (output layer)
    output_error = self.expected_output - self.output_layer
    error_output_layer = output_error * self.sigmoid_prime(self.output_layer)
    cost_of_output_weights = np.dot(self.hidden_layers[hidden_layer_len - 1].T, error_output_layer)

    # 2b. Get cost deriv. of weights: (hidden layer -2) <-- (hidden layer -1)
    costs_of_hidden_weights = {}
    for layer in range(hidden_layer_len - 1, 0, -1): # n..1
        prev_layer = self.hidden_layers[layer - 1]
        cur_layer = self.hidden_layers[layer]
        cur_weights = self.hidden_weights[layer]
        this_layers_err = self.expected_output - cur_layer
        error_hidden_layer = this_layers_err * self.sigmoid_prime(cur_layer)
        costs_of_hidden_weights[layer] = np.dot( prev_layer.T, np.dot(error_hidden_layer, cur_weights.T) * self.sigmoid_prime(cur_layer) * self.sigmoid_prime(prev_layer) )

    # 2c. Get cost deriv. of weights: (input layer) <-- (first hidden layer)
    input_error = self.expected_output - self.input_layer
    error_input_layer = input_error * self.sigmoid_prime(self.input_layer)
    cost_of_input_weights = np.dot(self.input_layer.T, np.dot(error_input_layer, self.hidden_weights[0]) * self.sigmoid_prime(self.hidden_layers[0]))

    # 2d. Update weight matricies
    self.hidden_weights[0] += cost_of_input_weights * self.learning_rate
    for key, value in costs_of_hidden_weights.items():
        self.hidden_weights[key] += value * self.learning_rate
    self.output_weights += cost_of_output_weights * self.learning_rate

    # Debug
    if (self.is_debugging):
      for layer in range(hidden_layer_len):
         print("`hidden_layer_{}`\n {}\n".format(layer, self.hidden_layers[layer]))
      print("`output_layer`\n {}\n".format(self.output_layer))
      print("`output_error`\n {}\n".format(output_error))
      print("`cost_of_output_weights, layer {}\n` {}\n".format(hidden_layer_len, cost_of_output_weights.tolist()))
      for key, value in costs_of_hidden_weights.items():
          print("costs_of_hidden_weights, layer", key, "\n", value, "\n")
      print("`cost_of_input_weights, layer {}\n` {}\n".format(hidden_layer_len, cost_of_input_weights.tolist()))

  def output_results(self):
    return self.output_layer
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def sigmoid_prime(self, sigmoid_of_z):
    return sigmoid_of_z * (1 - sigmoid_of_z)


### 

a_b_in = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
q_out = np.array([[0], [1], [1], [0]])
net = XORDeepNeuralNet(
  inputs=a_b_in, 
  outputs=q_out,
  is_debugging=False)

net.__call__()

outer_n = 10
iters_per_n = 2500
history = {}

for x in range(0, outer_n):
  n = iters_per_n * (x + 1)
  for _ in range(0, n):
    net.__call__()
  error = round(np.mean(np.abs(np.subtract(net.output_results(), q_out))) * 100, 6)
  history[n] = error
  print("After {} iterations . . .\n----------------------".format(n))
  print("model output = ({})".format(net.output_results().tolist()))
  print("desired output = ({})".format(q_out.tolist()))
  print("error = {}%\n\n".format(error))

endTime = datetime.now()
print("\n* * * *  END  CODE EXECUTION * * * *\n")
print("Ended script at: {}".format(endTime))
print("Script execution time: {}\n".format(endTime - startTime))

lists = sorted(history.items())
x, y = zip(*lists)
plt.plot(x, y)
plt.xlabel('# of Iterations')
plt.ylabel('Error %')
plt.title('XOR NN Prediction Performance - VARIABLE LAYER')
plt.show()