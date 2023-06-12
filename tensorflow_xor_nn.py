from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

# START CODE EXECUTION TIMER
startTime = datetime.now()
print("\nStarted script at: {}".format(startTime))
print("\n* * * * BEGIN CODE EXECUTION * * * *\n")

### XOR GATE IMPLEMENTED IN TENSORFLOW 

xor_in = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_out = np.array([[0], [1], [1], [0]])

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(2,)), # IN
    tf.keras.layers.Dense(4, activation=tf.keras.activations.sigmoid), # HIDDEN
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid) # OUT
])

# Compile the model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.25),
  loss=tf.keras.losses.MeanSquaredError(),
  metrics=['mse', 'binary_accuracy'])

# Train the model
history = model.fit(
    xor_in,
    xor_out,
    epochs=2500,
    verbose=0)

# Predictions w/ error %
predictions = model.predict_on_batch(xor_in)
print(predictions)
print("\nError %\n", np.average(predictions - xor_out) * 100, "%\n")

###

# END CODE EXECUTION TIMER
print("\n* * * *  END  CODE EXECUTION * * * *\n")
endTime = datetime.now()
print("Ended script at: {}".format(endTime))
print("Script execution time: {}\n".format(endTime - startTime))

# Calculate the error percentage values over the epochs
error_percentage = [(1 - accuracy) * 100 for accuracy in history.history['binary_accuracy']]

# Plot the error percentage values over the epochs
plt.plot(error_percentage)
plt.xlabel('# of Iterations')
plt.ylabel('Error %')
plt.title('XOR NN Prediction Performance - TENSORFLOW')
plt.show()