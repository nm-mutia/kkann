import numpy as np
#import matplotlib.pyplot as plt

def print_network(net):
    for i, layer in enumerate(net, 1):
        print("Layer {} ".format(i))
        for j, neuron in enumerate(layer, 1):
            print("neuron {} :".format(j), neuron)


def initialize_network(X,outputSize):
    input_neurons = len(X[0])
    hidden_neurons = input_neurons + 1
    output_neurons = outputSize

    n_hidden_layers = 1

    net = list()

    for h in range(n_hidden_layers):
        if h != 0:
            input_neurons = len(net[-1])

        hidden_layer = [{'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons)]
        net.append(hidden_layer)

    output_layer = [{'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)

    return net

def activate_sigmoid(sum):
    return (1 / (1 + np.exp(-sum)))


def forward_propagation(net, input):
    row = input
    for layer in net:
        prev_input = np.array([])
        for neuron in layer:
            sum = neuron['weights'].T.dot(row)

            result = activate_sigmoid(sum)
            neuron['result'] = result

            prev_input = np.append(prev_input, [result])
        row = prev_input

    return row


def sigmoidDerivative(output):
    return output * (1.0 - output)


def back_propagation(net, row, expected):
    for i in reversed(range(len(net))):
        layer = net[i]
        errors = np.array([])
        if i == len(net) - 1:
            results = [neuron['result'] for neuron in layer]
            errors = expected - np.array(results)
        else:
            for j in range(len(layer)):
                herror = 0
                nextlayer = net[i + 1]
                for neuron in nextlayer:
                    herror += (neuron['weights'][j] * neuron['delta'])
                errors = np.append(errors, [herror])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoidDerivative(neuron['result'])


def updateWeights(net, input, lrate):
    for i in range(len(net)):
        inputs = input
        if i != 0:
            inputs = [neuron['result'] for neuron in net[i - 1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]


def training(net, epochs, lrate, n_outputs):
    errors = []
    for epoch in range(epochs):
        sum_error = 0
        for i, row in enumerate(X):
            outputs = forward_propagation(net, row)

            expected = [0.0 for i in range(n_outputs)]
            expected[y[i]] = 1

            sum_error += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))])
            back_propagation(net, row, expected)
            updateWeights(net, row, 0.05)
        if epoch :
            print('>epoch=%d,error=%.3f' % (epoch, sum_error))
            errors.append(sum_error)
    return errors


inpdata = input("Pilih \n1.XOR 2.OR 3.AND 4.ParityBit :")
inpdata = int(inpdata)
inputSize = input("Panjang Input : ")
inputSize = int(inputSize)
inx = input("Jumlah Input : ")
inx = int(inx)
inputx = []
inx2 = 0

for i in range(inx):
    for j in range(inputSize):
        print("[", i, "][", j, "] = ")
        inputx.append(int(input(inx2)))

#inputx=np.array(inputx)
print(inputx)

outputSize = input("Jumlah Output : ")
outputSize = int(outputSize)

lrate=input("Learning Rate : ")
lrate = float(lrate)

if inpdata == 1:
    # Xor data
    XORdata = np.array([inputx])
    X = XORdata[:, 0:2]
    y = XORdata[:, -1]
    net = initialize_network(X,outputSize)
    print_network(net)
    errors = training(net, 10, lrate, outputSize)

elif inpdata == 2:
    # or data
    ORdata = np.array([inputx])
    X = ORdata[:, 0:2]
    y = ORdata[:, -1]
    net = initialize_network(X,outputSize)
    print_network(net)
    errors = training(net, 10, lrate, outputSize)

elif inpdata == 3:
    # and data [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
    ANDdata = np.array([inputx])
    X = ANDdata[:, 0:2]
    y = ANDdata[:, -1]
    net = initialize_network(X,outputSize)
    print_network(net)
    errors = training(net, 10, lrate, outputSize)

elif inpdata == 4:
    PARdata = np.array([inputx])
    X = PARdata[:, 0:2]
    y = PARdata[:, -1]
    net = initialize_network(X,outputSize)
    print_network(net)
    errors = training(net, 10, lrate, outputSize)


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagation(net, row)
    return outputs


pred = predict(net, np.array([1, 0]))
output = np.argmax(pred)
print(output)

print_network(net)