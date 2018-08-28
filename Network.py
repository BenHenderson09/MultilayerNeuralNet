import numpy as np

# --- Hyper Parameters ---

# The learning rate is used to effect how much we change our weights in backpropagation.
learningRate = 0.1

# The iterations is how many times we adjust the weights of our network
iterations = 70000

# The hidden layer nodes is how many nodes within the hidden layer
hiddenLayerNodes = 10

# --- Input training examples (hours slept, hours studied, iq) --- 
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
            
# --- Expected output for each training example ---
y = np.array([[0],
			[1],
			[1],
			[0]])

# --- Nonlinear activation function using sigmoid ---
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Finding the derivative of the activation function (mathematic-heavy code will look less messy using a function)
def sigDeriv(x):
    return x*(1-x)

# Setting the seed means we will use the same random number method throughtout our network. This makes random values more predictable.
# This means all of our weights will still be random, but there will be a correlation between them. Their values will be similar.
np.random.seed(1)

# Here we are setting our weight matrices (2 dimensional arrays), to a randomly seeded value with a mean of 0.
# The reason we give them a mean of 0 is similar to the reason we set the seed. This is to make similar.
# Note: X[0].size denotes the number of input nodes.
weights0 = 2 * np.random.random((X[0].size,hiddenLayerNodes)) - 1
weights1 = 2 * np.random.random((hiddenLayerNodes,X[0].size)) - 1

# This is where the network is trained. The number of iterations is set to a high value because weights are altered by a small value
# each iteration. The higher value of iterations means more precise predictions, but once it reaches a certain point there is very
# little change in our error. Setting it to 70,000 iterations will produce roughly the same results as setting it 700,000 iterations.
for i in range(iterations):

    # --- Feed forward and make our layers ---

    # layer 0 is simply our input layer
    layer0 = X

    # layer 1 is our hidden layer and layer2 is our output layer. The values of each node
    # in these layers is set to the previous layer's values multiplied
    # by the weights connecting to that layer using the dot product function. This is then put
    # through an activation function, in our case it is the sigmoid activation.
    layer1 = sigmoid(np.dot(layer0, weights0))
    layer2 = sigmoid(np.dot(layer1, weights1))

    # --- Backpropagation ---

    # The total error of our network is set to the expected output minus the network's output.
    # Remember that we are using examples here, to find out how wrong our network is, we just need
    # to find the difference between it's output and the predifined output (y).
    error = layer2-y


    # --- Calculus starts here ---
    
    # Here we are finding the delta of layer 2. This is the chain rule being applied to 
    # the function we used to actually make layer2. 
    # layer2 = sigmoid(np.dot(l1,weights1)) : derivative of outer function = sigDeriv(layer2) or sigmoid(np.dot(layer1,weights1)) * (1-sigmoid(np.dot(layer1,weights1)))
    # the reason we only derive the outer function is because we will later find the inner functions of layer1 and weights1
    layer2Delta = error * sigDeriv(layer2)

    # Here we are getting the derivative of weights1. Weights1 are the weights connected to layer2, the forward propagation
    # of layer2 = sigmoid(np.dot(layer1,weights1)). So to find the derivative we would use the chain rule, first find the derivative
    # of the outer expression and then multiply it by the inner expression. We have already calculated the outer expression,
    # it is layer2Delta. Now we just need the inner expression. This equals to be simply layer1.
    weights1Deriv = np.dot(layer1.T, layer2Delta)

    # Here we are finding how much layer1 contributed to the total error of our network.
    # Finding the derivative here is very similar to finding the derivative of weights1, the only difference being
    # that the in the inner function (np.dot(layer1,weights1)), the derivative of layer1 is simply weights1.
    layer1Error = np.dot(layer2Delta, weights1.T)

    # Here we are finding the delta of layer1. This is very similar to finding the delta of layer2,
    # we are just finding the derivative of the outer expression so we can use it to calculate the derivative
    # of weights0. The creating layer1 look like this, layer1 = sigmoid(np.dot(layer0, weights0)).
    # The derivative of the outer function is just sigDeriv(layer1) or sigmoid(np.dot(layer0,weights0)) * (1-sigmoid(np.dot(layer0,weights0)))
    layer1Delta = layer1Error * sigDeriv(layer1)

    # Here we are getting the derivative of weights0. This is again similar to finding the derivative of weights1,
    # we just need to multiply the derivative of the outer expression (layer1Delta) by the derivative of the inner expression.
    # The inner expression = np.dot(layer0, weights0). The derivative of weights0 is simply layer0.
    weights0Deriv = np.dot(layer0.T, layer1Delta)

    # Here we are using gradient descent to adjust our weights. We are subtracting the learning rate multiplied by the derivatives of the weights
    # to make the error of the network lower. Ultimately a derivative is a slope, the derivative of the weights can be
    # thought of as the slope with respect to the error. If we graphed the error with different weight values, you would
    # notice that at a point in the graph the line would be lowest on the Y-axis. This is where the weight causes the lowest error.
    # We take advantage of this by calculating the derivative (slope) of the weight at the point in the graph. If the slope is positive 
    # we need to decrease the weight, if the slope is negative we need to increase the weight.
    #
    # This means that our weight will find the lowest point on the Y axis, with the lowest error.
    # This is how we find the best number to set our weight, to get a lower error. This is the foundation of how a neural network works.
    #
    # Note that we always decrease the weight by the derivative. This is because if the derivative is positive,
    # it will be subtracted from the weight and if it is negative it will be
    # added. Remember that subtracting a negative means to just add.

    weights1 -= learningRate*weights1Deriv
    weights0 -= learningRate*weights0Deriv

    # Print the error at the start of training and then once every 10,000 iterations
    if ( i % (iterations/10) == 0 or i == 0 ):
        print("\nNetwork Error: ", np.mean(error))

    # When the network is finished training, predict some new data
    if (i == iterations-1 ):
        unknown = np.array([[1, 1, 1]])
        predict(unknown)
    
    def predict(x):
        # Forward propagating our new data through the network
        hidden     = sigmoid(np.dot(x, weights0))
        output     = sigmoid(np.dot(hidden, weights1))
        prediction = int(np.mean(np.around(output)))

        # Calculate the probability that our prediction is correct
        if (prediction == 0):
            chance = 100 - (100 * np.mean(output))
        else:
            chance = 100 * np.mean(output)
    
        print("\n---------------------------------------------------------")
        print("\nData: ", x)
        print("\nPrediction: ", prediction )
        print("\nProbability: ", chance,"%")





