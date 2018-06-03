#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from random import random

class Neuron(object):
    def __init__(self):
        self.weights = []
        self.bias = -1 + random()*2
        self.input = []
        self.output = 0
        self.delta = 0

    def calc_activation(self, arg_output):
        #sigmoid
        return 1.0/(1.0 + np.exp(-arg_output))

    def init_inputs(self, arg_n):
        for n in range(arg_n):
            self.weights.append(-1 + random()*2)
            self.input.append(0)

class Layer(object):
    def __init__(self, arg_n):
        self.neurons = []

        for n in range(arg_n):
            self.neurons.append(Neuron())

class Network(object):
    def __init__(self, arg_input, arg_hidden, arg_output):
        self.hidden_layers = []

        input_layer = arg_input

        for n_hidden in arg_hidden:            
            hl = Layer(n_hidden)
            for neuron in hl.neurons:
                neuron.init_inputs(input_layer)

            self.hidden_layers.append(hl)

            input_layer = len(hl.neurons)

        self.output_layer = Layer(arg_output)

        for neuron in self.output_layer.neurons:
            neuron.init_inputs(len(self.hidden_layers[-1].neurons))

    def forward_pass(self, arg_input):
        input = arg_input
        output = []

        for layer in self.hidden_layers:
            new_inputs = []
            for neuron in layer.neurons:
                neuron.input = input
                neuron.output = neuron.bias + np.dot(neuron.input, neuron.weights)
                neuron.output = neuron.calc_activation(neuron.output)
                new_inputs.append(neuron.output)

            input = new_inputs

        for neuron in self.output_layer.neurons:
            neuron.input = input
            neuron.output = neuron.bias + np.dot(neuron.input, neuron.weights)
            neuron.output = neuron.calc_activation(neuron.output)
            output.append(neuron.output)

        return output
            
    def backprop(self, arg_input, arg_output, arg_rate=0.1):
        target_output = arg_output
        self.forward_pass(arg_input)

        #output layer
        for neuron, target_output in zip(self.output_layer.neurons, arg_output):
            neuron.delta = (target_output - neuron.output) * neuron.output * (1 - neuron.output)            
            new_weights = []
            for w, x in zip(neuron.weights, neuron.input):                
                w += arg_rate * neuron.delta * x               
                new_weights.append(w)

            neuron.weights = new_weights

            # bias
            neuron.bias += arg_rate * neuron.delta

        downstream = self.output_layer.neurons

        #hidden layers back to front
        for layer in reversed(self.hidden_layers):
            for i, neuron in enumerate(layer.neurons):
                delta = neuron.output * (1 - neuron.output) 

                #downstream
                sum_downstream = 0
                for neuron_down in downstream:
                    sum_downstream += neuron_down.delta * neuron_down.weights[i]

                neuron.delta = delta * sum_downstream
                new_weights = []
                for w, x in zip(neuron.weights, neuron.input):                   
                    w += arg_rate * neuron.delta * x
                    new_weights.append(w)

                neuron.weights = new_weights

                #bias
                neuron.bias += arg_rate * neuron.delta

            downstream = layer.neurons

    def train(self, arg_inputs, arg_output, arg_n = 100):
        errors = []

        for n in range(arg_n):
            error = 0
            for input, output in zip(arg_inputs, arg_output):
                self.backprop(input, output)
                error += self.calc_error(output, self.forward_pass(input))

            errors.append(error)

        return errors
            

    def calc_error(self, arg_target_output, arg_output):
        error = 0

        for target, out in zip(arg_target_output, arg_output):
            error += np.square(target - out)

        return error/2

def main():
    network = Network(2, [10,10], 1)

    #xor
    input = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output = [[0], [1], [1], [0]]

    errors = network.train(input, output, 10000)

    for i in input:
        network.train(input, output, 1)
        print(i, network.forward_pass(i))

    plt.plot(errors)
    plt.show()

if __name__=="__main__":
    main()