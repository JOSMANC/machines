import matplotlib.pyplot as plt
import numpy as np
import neural
import itertools
import math

least_squares = 'output - target'
tanh = 'math.tanh(x)'
tanh_prime = '1 - math.tanh(x)**2'
net = neural.network({'inputs': 1,
                      'cost_prime': least_squares,
                      'layers': [{'activation': tanh,
                                  'gradient': tanh_prime,
                                  'nodes': 30},
                                 {'activation': 'x',
                                  'gradient': '1',
                                  'nodes': 1}]})

magnitude = 2
steps = 100
learning_rate = .02
X = np.linspace(-magnitude, magnitude, steps)

predictions = [list(itertools.chain.from_iterable(net.predict(X)))]
sin = np.vectorize(math.sin)

for epoch in range(5):
    print('.')
    for i in range(200):
        signal = np.random.randn(100) * magnitude
        target = signal**3
        net.learn(signal, target, learning_rate)
    predictions.append(list(itertools.chain.from_iterable(net.predict(X))))

f, axis = plt.subplots()
for p in predictions:
    axis.plot(X, p)

plt.show()
