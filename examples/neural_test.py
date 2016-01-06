import matplotlib.pyplot as plt
import numpy as np
import neural
import itertools
import math
import json

least_squares = 'output - target'
tanh = 'math.tanh(x)'
tanh_prime = '1 - math.tanh(x)**2'
net = neural.network({'inputs': 1,
                      'cost_prime': least_squares,
                      'layers': [{'activation': tanh,
                                  'gradient': tanh_prime,
                                  'nodes': 40},
                                 {'activation': 'x',
                                  'gradient': '1',
                                  'nodes': 1}]})

magnitude = 1
steps = 100
learning_rate = .1
X = np.linspace(-magnitude, magnitude, steps)

predictions = [list(itertools.chain.from_iterable(net.predict(X)))]
sin = np.vectorize(math.sin)

for epoch in range(10):
    print('.')
    for i in range(50):
        signal = np.random.randn(100) * magnitude
        target = signal**2
        net.learn(signal, target, learning_rate)
    predictions.append(list(itertools.chain.from_iterable(net.predict(X))))

f, axis = plt.subplots()
for p in predictions:
    axis.plot(X, p)

plt.show()

# with open('test.txt', 'w') as outfile:
#     json.dump(net.save_json(), outfile)
