import numpy as np
from itertools import count
import math


def descend(positions, distances, rate):
    pos_delta = np.zeros_like(positions)
    total_energy = 0

    for distance in distances:
        pos_0 = positions[distance[0]]
        pos_1 = positions[distance[1]]

        d = distance[2]
        s = 0
        x_delta = np.zeros_like(pos_0)

        for idx, x0, x1 in zip(count(), pos_0, pos_1):
            s += (x0 - x1)**2
            x_delta[idx] = x0 - x1

        if s <= 0:
            s = 0.1
        if d <= 0:
            d = 0.1
        s = math.sqrt(s)
        x_delta = (d-s)/(d*s) * x_delta
        pos_delta[distance[0]] += rate * x_delta
        pos_delta[distance[1]] -= rate * x_delta

        total_energy += (d-s)**2 / d

    return [positions + pos_delta, total_energy]


def generate_map(ids, distances, dims):
    dists = [[ids.index(d[0]), ids.index(d[1]), abs(d[2])] for d in distances]
    positions = np.random.randn(len(ids), dims)

    prev_energy = 0
    rate_multiplier = 1
    for i in range(2000):
        positions, energy = descend(positions, dists, .025 * rate_multiplier)
        rate_multiplier = 1 + 10/(0.1 + (prev_energy - energy)**2)
        prev_energy = energy
        print(i, energy, rate_multiplier)

    return [[ids[i], p.tolist()] for i, p in zip(count(), positions)]

"""
ids = ["cake", "pie", "muffin"]

distances = [["cake", "pie", 2.5],
             ["pie", "cake", 2.5]]

test = generate_map(ids, distances, 1)
print(test)
"""

"""
pos = np.array([[-1.2, 1.9, 0.0], [0.0, 1.0, -8.1], [1.4, -7.2, 1.1]])
dist = np.array([[0, 1, 1.0], [1, 2, 3.4], [0, 2, 2.3]])

for i in range(50):
    pos, energy = descend(pos, dist, 0.5)
    print(pos)
    print(energy)
    print("------")
"""
