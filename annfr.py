import numpy as np
#import matplotlib.pyplot as plt

def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)


def initialize_network(X, outputSize, hidl):
    input_neurons = len(X[0])
    hidden_neurons = hidl #input_neurons + 1
    output_neurons = outputSize #2

    n_hidden_layers = 1

    net = list()

    for h in range(n_hidden_layers):
        if h != 0:
            input_neurons = len(net[-1])

        hidden_layer = [{'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons)]
        net.append(hidden_layer)
    
    if hidl > 0:
        output_layer = [{'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
        net.append(output_layer)
    elif hidl == 0:
        output_layer = [{'weights': np.random.uniform(size=input_neurons)} for i in range(output_neurons)]
        net.append(output_layer)
    
    return net

def activate(weights, inputs):
    activation = weights[-1]  # Added the neuron bias beforehand. -1 is used for last element of an array.
    for i in range(len(weights) - 1):  # Taking only first two weights as third one is the bias, which has been alredy added.
        activation += weights[i] * inputs[i]  # Calculating the activation of the neuron
    return activation - threshold


def activate_step(sumi):
    if (sumi >= threshold).all():
        return 1
    else:
        return 0

def forward_propagation(net, inputhai):
    row = inputhai

    for layer in net:
        #if hidl == 0:
         #    continue
        prev_input = np.array([])
        for neuron in layer:
            sumi = neuron['weights'].T.dot(row)
            sumi = sumi - threshold
            #activation = activate(neuron['weights'], row)
            result = activate_step(sumi)
            neuron['result'] = result

            prev_input = np.append(prev_input, [result])
        row = prev_input

    return row


def stepDerivative(output):
    return output*(1.0-output)


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
            #neuron['delta'] = errors[j] * stepDerivative(neuron['result'])
            neuron['delta'] = y[j] - neuron['result'] 

def updateWeights(net, inputhai, lrate):
    for i in range(len(net)):
        inputs = inputhai
        if i != 0:
            inputs = [neuron['result'] for neuron in net[i - 1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagation(network, row)
    return outputs

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
            updateWeights(net, row, lrate)
        #if epoch:
        print('>epoch=%d, error=%.3f' % (epoch, sum_error))
        errors.append(sum_error)
        for j, row in enumerate(X):
            prediction = predict(net, row)
            print('Desired output=%d, Actual output=%d' % (y[j], np.argmax(prediction)))
        if sum_error == 0:
            break
    return errors

inpdata = input("Choose \n1.XOR 2.OR 3.AND 4.ParityBit : ")
inpdata = int(inpdata)
inputSize = input("Length per-input : ")
inputSize = int(inputSize)
inx = input("Jumlah Input : ")
inx = int(inx)
inputx = []

for i in range(inx):
    row_list=[]
    for j in range(inputSize):
        print("[",i,"][",j,"] =")
        row_list.append(int(input()))
    inputx.append(row_list)
    print(inputx[i])

print(inputx)
       
outputSize = int(input("Jumlah Output : "))
lrate = float(input("Learning Rate : "))
threshold = float(input("Threshold : "))
hidl = int(input("Jumlah Hidden Neuron : "))
tester = []

print("Prediksi Input : ")
for i in range(inputSize-1):
    tester.append(int(input()))

if inpdata == 1:
    #Xor data
    XORdata = np.array(inputx)
    X = XORdata[:,0:inputSize-1]
    y = XORdata[:,-1]
    
elif inpdata == 2:
    #or data
    ORdata = np.array(inputx)
    X = ORdata[:,0:inputSize-1]
    y = ORdata[:,-1]
    
elif inpdata == 3:
    #and data [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
    ANDdata = np.array(inputx)
    X = ANDdata[:,0:inputSize-1]
    y = ANDdata[:,-1]
    
elif inpdata == 4:
    PARdata = np.array([inputx])
    X = PARdata[:,0:inputSize-1]
    y = PARdata[:,-1]

    
net = initialize_network(X,outputSize,hidl)
print_network(net)
#errors = 
training(net, 10, lrate, outputSize)


pred = predict(net,np.array(tester))
output = np.argmax(pred)
print("Output = ", output)

print_network(net)

#epochs = [0,1,2,3,4,5,6,7,8,9]
#plt.plot(epochs,errors)
#plt.xlabel("epochs in 10's")
#plt.ylabel('error')
#plt.show()