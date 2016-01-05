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

        s2 = 0
        x_delta = np.zeros_like(pos_0)

        for idx, x0, x1 in zip(count(), pos_0, pos_1):
            s2 += (x0 - x1)**2
            x_delta[idx] = x0 - x1

        s = math.sqrt(s2)
        x_delta = (d-s) * x_delta

        pos_delta[distance[0]] += rate * x_delta
        pos_delta[distance[1]] -= rate * x_delta

        total_energy += (d-s)**2

    return [positions + pos_delta, total_energy]


def embed(ids, distances, initial_positions, rate, iterations):
    dists = [[ids.index(d[0]), ids.index(d[1]), abs(d[2])] for d in distances]
    positions = initial_positions

    for i in range(iterations):
        positions, energy = descend(positions, dists, rate)
        print(i, energy)

    return [[ids[i], p] for i, p in zip(count(), positions)]


def generate_initial_positions(ids, dimensions):
    return np.random.randn(len(ids), dimensions)