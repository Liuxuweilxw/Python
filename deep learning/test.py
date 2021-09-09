import mnist_loader
import json
import network2
import numpy as np

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def feedforward(net, a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(net.biases, net.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a


def accuracy(net, data, convert=False):
    """Return the number of inputs in ``data`` for which the neural
    network outputs the correct result. The neural network's
    output is assumed to be the index of whichever neuron in the
    final layer has the highest activation.

    The flag ``convert`` should be set to False if the data set is
    validation or test data (the usual case), and to True if the
    data set is the training data. The need for this flag arises
    due to differences in the way the results ``y`` are
    represented in the different data sets.  In particular, it
    flags whether we need to convert between the different
    representations.  It may seem strange to use different
    representations for the different data sets.  Why not use the
    same representation for all three data sets?  It's done for
    efficiency reasons -- the program usually evaluates the cost
    on the training data and the accuracy on other data sets.
    These are different types of computations, and using different
    representations speeds things up.  More details on the
    representations can be found in
    mnist_loader.load_data_wrapper.

    """
    if convert:
        results = [(np.argmax(feedforward(net,x)), np.argmax(y))
                   for (x, y) in data]
    else:
        results = [(np.argmax(feedforward(net,x)), y)
                    for (x, y) in data]
    return sum(int(x == y) for (x, y) in results)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.load("C:/Users/liuxuwei/Desktop/out/a.txt")
count = accuracy(net,test_data)
print ("Accuracy on evaluation data: {} / {}".format(count, len(test_data)))
