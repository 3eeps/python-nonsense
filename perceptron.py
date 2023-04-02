# python-fun/perceptron.py

# imports
import numpy as np

# perceptron function
def perceptron(input_data , weights):
    activation = np.dot(input_data, weights)
    if activation > 0:
        return 1
    else:
        return 0

 # some training data with labels
training_data = np.array([[0.0], [0,1], [1,0], [1,1]])
labels = np.array([0, 0, 0, 1])

# init weights randomly
weights = np.random.rand(2)

# learning rate and number of epochs
learning_rate = 0.1
epochs = 100

# train the perceptron
for i in range(epochs):
    for j in range(len(training_data)):
        input_data = training_data[j]
        label = labels[j]
        prediction = perceptron(input_data, weights)
        error = label - prediction
        weights += learning_rate * error * input_data

# test the perceptron
test_data = np.array([[0.0], [0,1], [1,0], [1,1]])
for i in range(len(test_data)):
    input_data = test_data[i]
    prediction = perceptron(input_data, weights)
    print(f"input: {input_data}, prediction: {prediction}")
    
