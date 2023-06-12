from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# START CODE EXECUTION TIMER
startTime = datetime.now()
print("\nStarted script at: {}".format(startTime))
print("\n* * * * BEGIN CODE EXECUTION * * * *\n")

### XOR Self-Lab :)
###
### Solving a machine learning problem usually consists of the following steps:
###
### 1. Obtain training data.
### 2. Define the model.
### 3. Define a loss function.
### 4. Run through the training data, calculating loss from the ideal value
### 5. Calculate gradients for that loss and use an optimizer to adjust the variables to fit the data.
### 6. Evaluate your results.
###
### Given f(x) = x * W + b, where x is your input, f(x) is your output, W are your weights, and b is your bias. 
### The most basic machine learning problems involve a given x and y: find the slope and offset of a line via simple
### linear regression.
###
### XOR Truth Table:
###  A |  B |  Q
### ---------------
###  0 |  0 |  0
###  0 |  1 |  1
###  1 |  0 |  1
###  1 |  1 |  0
class XORNeuralNet:
  def __init__(self, inputs, outputs, hidden_layer_size=2, learning_rate=0.5, is_debugging=False, set_good_weights=False):
    super(XORNeuralNet, self).__init__()

    # Set our model-level variables
    self.inputs = inputs
    self.outputs = outputs
    self.is_debugging = is_debugging
    self.learning_rate = learning_rate

    # Number of neurons contained within each layer.
    self.INPUT_LAYER_SIZE = 2
    self.HIDDEN_LAYER_SIZE = hidden_layer_size
    self.OUTPUT_LAYER_SIZE = 1

    # Initialize our neural network items:
    if (set_good_weights == False):
      self.init_rand_weights()
    else:
      self.init_good_weights()
    self.init_zero_biases()
    self.init_layers()

    # Debug output
    if (is_debugging):
      print("\t> `XORNeuralNet` intitialized . . .")
      print("\t\t> `inputs` {}, shape{}".format(self.inputs.tolist(), self.inputs.shape))
      print("\t\t> `outputs` {}, shape{}".format(self.outputs.tolist(), self.outputs.shape))
      print("\t\t> `hidden_weights(rand)` {}, shape{}".format(self.hidden_weights.tolist(), self.hidden_weights.shape))
      print("\t\t> `output_weights(rand)` {}, shape{}".format(self.output_weights.tolist(), self.output_weights.shape))
      print("\t\t> `hidden_bias(zero)` {}, shape{}".format(self.hidden_bias.tolist(), self.hidden_bias.shape))
      print("\t\t> `output_bias(zero)` {}, shape{}".format(self.output_bias.tolist(), self.output_bias.shape))
      print("\t\t> `hidden_layer(empty)` {}, shape{}".format(self.hidden_layer.tolist(), self.hidden_layer.shape))
      print("\t\t> `output_layer(empty)` {}, shape{}\n".format(self.output_layer.tolist(), self.output_layer.shape))

  def __call__(self, *args: Any, **kwds: Any):
    if (self.is_debugging):
      print("\t> `__call__()` invoked with `args` {}, `kwds` {}".format(args, kwds))

    self.feed_foward()
    self.backpropagation()

  def output_results(self):
    return self.output_layer

  def output_hidden_weights(self):
    return self.hidden_weights
  
  def output_output_weights(self):
    return self.output_weights

  # GOOD WEIGHTS:
  # `hidden_weights` [[0.9671288381112517, 8.807774693218319], [0.9671301271237541, 8.809877669379418]]
  # `output_weights` [[-55.26049325500156], [44.315663025087325]]
  def init_good_weights(self):
    self.hidden_weights = np.array([[0.9671288381112517, 8.807774693218319], [0.9671301271237541, 8.809877669379418]])
    self.output_weights = np.array([[-55.26049325500156], [44.315663025087325]])

  def init_rand_weights(self):
    # np.random.randn(rows, cols)
    # The number of rows must equal the number of neurons in the previous layer. 
    # The number of columns must match the number of neurons in the next layer.
    self.hidden_weights = np.random.randn(self.INPUT_LAYER_SIZE, self.HIDDEN_LAYER_SIZE)
    self.output_weights = np.random.randn(self.HIDDEN_LAYER_SIZE, self.OUTPUT_LAYER_SIZE)

  def init_zero_biases(self):
    # Return a new array of given shape and type, filled with fill_value.
    # The shape of the hidden bias should be (1, hidden layer size)
    # The shape of the output bias should be (1, output layer size)
    # These are set to 0.0 as the XOR Gate does not require initial bias.
    self.hidden_bias = np.zeros((1, self.HIDDEN_LAYER_SIZE))
    self.output_bias = np.zeros((1, self.OUTPUT_LAYER_SIZE))
 
  def init_layers(self):
    # If the first matrix has dimensions m × n, 
    # and is multiplied by a second matrix of dimensions n × p, 
    # then the dimensions of the product matrix will be m × p.
    # The shape of the hidden layer should be (input(m) x hidden_weights(p))
    # The shape of the output layer should be (hidden_layer(m) x output_weights(p))
    m = self.inputs.shape[0]
    p = self.hidden_weights.shape[1]
    p_out = self.output_weights.shape[1]

    self.hidden_layer = np.zeros((m, p))
    self.output_layer = np.zeros((m, p_out))

  def backpropagation(self):
    # 1. Get error of the output layer => shape(4,1)
    model_err_raw_distance = self.outputs - self.output_layer # X - Q
    error_output_layer = model_err_raw_distance * self.sigmoid_prime(self.output_layer) # (X - Q) * sigmoid'(Z(q))

    # 2. Get error of the hidden layer => shape(4,2)
    out_err_against_weights = np.dot(error_output_layer, self.output_weights.T) # E(out) o T(Wo)
    error_hidden_layer = out_err_against_weights * self.sigmoid_prime(self.hidden_layer) # (E(out) o T(Wo)) * sigmoid'(Z(h))

    # 3. Get the cost derivative of the weights
    cost_of_output_weights = np.dot(self.hidden_layer.T, error_output_layer) # T(Z(h)) o E(out) => shape(2,1)
    cost_of_hidden_weights = np.dot(self.inputs.T, error_hidden_layer) # T(IN) o E(hidden) => shape(2,2)

    # 4. Update the weights
    self.output_weights += cost_of_output_weights * self.learning_rate
    self.hidden_weights += cost_of_hidden_weights * self.learning_rate

    if (self.is_debugging):
      print("\t\t> `backpropagation()` invoked . . .")
      print("\t\t\t> `model_err_raw_distance` {}, shape{}".format(model_err_raw_distance.tolist(), model_err_raw_distance.shape))
      print("\t\t\t> `error_output_layer` {}, shape{}".format(error_output_layer.tolist(), error_output_layer.shape))
      print("\t\t\t> `out_err_against_weights` {}, shape{}".format(out_err_against_weights.tolist(), out_err_against_weights.shape))
      print("\t\t\t> `error_hidden_layer` {}, shape{}".format(error_hidden_layer.tolist(), error_hidden_layer.shape))
      print("\t\t\t> `cost_of_output_weights` {}, shape{}".format(cost_of_output_weights.tolist(), cost_of_output_weights.shape))
      print("\t\t\t> `cost_of_hidden_weights` {}, shape{}".format(cost_of_hidden_weights.tolist(), cost_of_hidden_weights.shape))
      print("\t\t\t> `output_weights` {}, shape{}".format(self.output_weights.tolist(), self.output_weights.shape))
      print("\t\t\t> `hidden_weights` {}, shape{}".format(self.hidden_weights.tolist(), self.hidden_weights.shape))

  def feed_foward(self):
    self.hidden_layer = self.sigmoid(np.dot(self.inputs, self.hidden_weights) + self.hidden_bias)
    self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.output_weights) + self.output_bias)
    
    # Debug output
    if (self.is_debugging):
      print("\t\t> `feed_foward()` invoked . . .")
      print("\t\t\t> `hidden_layer*` {}".format(self.hidden_layer.tolist()))
      print("\t\t\t> `output_layer*` {}".format(self.output_layer.tolist()))

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def sigmoid_prime(self, sigmoid_of_z):
    return sigmoid_of_z * (1 - sigmoid_of_z)

# Let's try this now! :)
# Inputting XOR truth table as A|B in and Q out
a_b_in = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
q_out = np.array([[0], [1], [1], [0]])
net = XORNeuralNet(
  inputs=a_b_in, 
  outputs=q_out,
  hidden_layer_size=4,
  learning_rate=1.5,
  is_debugging=False,
  set_good_weights=False)

# Intialize our vars and perform!
outer_n = 10
iters_per_n = 2500
history = {}

for x in range(0, outer_n):
  n = iters_per_n * (x + 1)
  for _ in range(0, n):
    net.__call__()
  print("After {} iterations . . .\n----------------------".format(n))
  print("model output = ({})".format(net.output_results().tolist()))
  print("desired output = ({})".format(q_out.tolist()))
  error = round(np.mean(np.abs(np.subtract(net.output_results(), q_out))) * 100, 6)
  print("error = {}%\n\n".format(error))
  history[n] = error

# Output our final weights we arrived at
print("\nFINAL:\n")
print("`hidden_weights` {}".format(net.output_hidden_weights().tolist()))
print("`output_weights` {}".format(net.output_output_weights().tolist()))

# END CODE EXECUTION TIMER
print("\n* * * *  END  CODE EXECUTION * * * *\n")
endTime = datetime.now()
print("Ended script at: {}".format(endTime))
print("Script execution time: {}\n".format(endTime - startTime))

# Plot our historical error percentage against iteration #
lists = sorted(history.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.xlabel('# of Iterations')
plt.ylabel('Error %')
plt.title('XOR NN Prediction Performance - SINGLE LAYER')
plt.show()