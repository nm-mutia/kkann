from random import random
import numpy as np

# initialising a neural network
def init_network(inputs, hid, outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(inputs + 1)]} for i in range(hid)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(hid + 1)]} for i in range(outputs)]
    network.append(output_layer)
    return network

def activate(weights, inputs):
    activation = weights[-1]  # Added the neuron bias beforehand. -1 is used for last element of an array.
    for i in range(len(weights) - 1):  # Taking only first two weights as third one is the bias, which has been alredy added.
        activation += weights[i] * inputs[i]  # Calculating the activation of the neuron
    return activation - threshold


def transfer(activation):
    if activation >= 0:
        return 1
    else: return 0

def forward_prop(network,row):  # Forward Propagation....calculating the transfer activity of neuron and passing it to the neurons in next layer.
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# network = init_network(2,1,2)
# row = [1,0,None]
# output = forward_prop(network, row)
# print(output)

def transfer_der(output):
    return output * (1 - output)


def backward_prop_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_der(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]  #
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]  # Updating the weights of a neuron.
            neuron['weights'][-1] += l_rate * neuron['delta']  # Updating the bias of a neuron

# Make a prediction with a network
def predict(network, row):
 outputs = forward_prop(network, row)
 return outputs.index(max(outputs))

def train_network(network, train, l_rate, n_outputs):
    n_epoch = 999
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_prop(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_prop_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch+1, l_rate, sum_error))
        for row in dataset:
            prediction = predict(network, row)
            print('Expected=%d, Got=%d' % (row[-1], prediction))
        if sum_error == 0 :
            break

inpdata = input("Pilih \n1.XOR 2.OR 3.AND 4.ParityBit : ")
inpdata = int(inpdata)
outputSize = input("Jumlah Output : ")
outputSize = int(outputSize)

lrate = input("Learning Rate : ")
lrate = float(lrate)

if inpdata == 1:
    # Xor data
    dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    n_inputs = len(dataset[0]) - 1  # Third value in the dataset entry is for biases.
    n_outputs = len(set(row[-1] for row in dataset))

elif inpdata == 2:
    # or data
    dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    n_inputs = len(dataset[0]) - 1  # Third value in the dataset entry is for biases.
    n_outputs = len(set(row[-1] for row in dataset))

elif inpdata == 3:
    # and data [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
    dataset = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]
    n_inputs = len(dataset[0]) - 1  # Third value in the dataset entry is for biases.
    n_outputs = len(set(row[-1] for row in dataset))

elif inpdata == 4:
    dataset = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]


hidden_l = int(input('Jumlah hidden Layer : '))
threshold = float(input('Threshold : '))
network = init_network(n_inputs, hidden_l, n_outputs)
train_network(network, dataset, lrate, n_outputs)

for layer in network:
    print(layer)