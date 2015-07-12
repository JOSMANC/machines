import numpy as np
import math


class network:
    def __init__(self, json):
        self.weights = []
        self.bias = []
        self.activation = []
        self.gradient = []
        self.actv_source = []
        self.grad_source = []

        prev_size = json['inputs']
        for layer in json['layers']:
            size = layer['nodes']
            actv = eval("lambda x:" + layer['activation'])
            grad = eval("lambda x:" + layer['gradient'])

            self.actv_source.append(layer['activation'])
            self.grad_source.append(layer['gradient'])
            self.activation.append(np.vectorize(actv))
            self.gradient.append(np.vectorize(grad))
            self.weights.append(np.random.randn(size, prev_size) / prev_size)
            self.bias.append(np.random.randn(size, 1))

            prev_size = size

    @staticmethod
    def load_json(json):
        net = network({'inputs': 0, 'layers': []})
        for layer in json:
            actv = eval("lambda x:" + layer['actv'])
            grad = eval("lambda x:" + layer['grad'])

            net.actv_source.append(layer['actv'])
            net.grad_source.append(layer['grad'])
            net.activation.append(np.vectorize(actv))
            net.gradient.append(np.vectorize(grad))
            net.weights.append(np.array(layer['weights']))
            net.bias.append(np.array(layer['bias']))

        return net

    def save_json(self):
        json = []
        for a, g, w, b in zip(self.actv_source,
                              self.grad_source,
                              [x.tolist() for x in self.weights],
                              [x.tolist() for x in self.bias]):

            json.append({'actv': a, 'grad': g, 'weights': w, 'bias': b})

        return json

    def predict(self, signals):
        output = []
        for signal in signals:
            signal = np.array(signal, ndmin=2)
            for actv, b, w in zip(self.activation, self.bias, self.weights):
                signal = actv(w.dot(signal) + b)

            output.append(signal)

        return output

    def learn(self, inputs, targets, learning_rate=0.05):
        weight_delta = [0 * x for x in self.weights]
        bias_delta = [0 * x for x in self.bias]

        # We need to loop through every signal and target in the batch
        for signal, target in zip(inputs, targets):
            signal = np.array(signal, ndmin=2)
            target = np.array(target, ndmin=2)

            gradients = []
            signals = []

            # Feed each signal through the network, we need to record the
            # output and gradient of each layer since these are needed in the
            # backprop algorithm
            for actv, grad, w, b in zip(self.activation,
                                        self.gradient,
                                        self.weights,
                                        self.bias):
                signals.append(signal)

                argument = w.dot(signal) + b
                signal = actv(argument)
                gradient = grad(argument)

                gradients.append(gradient)

            backprop_values = zip(reversed(self.weights),
                                  reversed(gradients),
                                  reversed(signals),
                                  reversed(weight_delta),
                                  reversed(bias_delta))

            # We loop backwards through the net, propagating the error and
            # saving our weight and bias corrections
            error = (signal - target)

            for w, g, s, wd, bd in backprop_values:
                error = error * g
                bd += error
                wd += np.outer(error, s)
                error = w.T.dot(error)

        # After processing the whole batch we adjust our weights and biases
        scale = len(inputs)

        self.weights = [w - learning_rate * delta / scale
                        for w, delta in zip(self.weights, weight_delta)]

        self.bias = [b - learning_rate * delta / scale
                     for b, delta in zip(self.bias, bias_delta)]
