import matplotlib.pyplot as plt
import numpy as np
import neural
import itertools
import math

net = neural.network({'inputs': 1,
                      'layers': [{'activation': 'math.tanh(x)',
                                  'gradient': '1 - math.tanh(x)**2',
                                  'nodes': 40},
                                 {'activation': 'math.tanh(x)',
                                  'gradient': '1 - math.tanh(x)**2',
                                  'nodes': 40},
                                 {'activation': 'x',
                                  'gradient': '1',
                                  'nodes': 1}]})

magnitude = 1
steps = 100
learning_rate = .05
X = np.linspace(-magnitude, magnitude, steps)

predictions = [list(itertools.chain.from_iterable(net.predict(X)))]

for epoch in range(10):
    for i in range(50):
        signal = np.random.randn(10) * magnitude
        target = signal**2
        net.learn(signal, target, learning_rate)
    predictions.append(list(itertools.chain.from_iterable(net.predict(X))))

f, axis = plt.subplots()
for p in predictions:
    axis.plot(X, p)

plt.show()
