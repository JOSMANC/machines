import numpy as np
from itertools import count
import math


def descend(positions, distances, rate):
    # cumulatively store the changes in position due to each distance pair
    pos_delta = np.zeros_like(positions)
    total_energy = 0

    for distance in distances:
        # here d is target distance, s is current distance for this pair
        pos_0 = positions[distance[0]]
        pos_1 = positions[distance[1]]
        d = distance[2]
        s2 = 0
        x_delta = np.zeros_like(pos_0)

        # calculate the gradient of our energy: sum of (d-s)^2 over all pairs
        for idx, x0, x1 in zip(count(), pos_0, pos_1):
            s2 += (x0 - x1)**2
            x_delta[idx] = x0 - x1

        s = math.sqrt(s2)
        grad = (d-s) * x_delta

        # update our position deltas based on the gradient and descent rate
        pos_delta[distance[0]] += rate * grad
        pos_delta[distance[1]] -= rate * grad
        total_energy += (d-s)**2

    # update positions and return the new positions and total energy
    return [positions + pos_delta, total_energy]


def embed(ids, distances, initial_positions, rate, iterations):
    # Since the distance list length generally grows as the square of the
    # ids list length it can be very intensive to generate the complete ids list
    # from the distances list. For this reason we ask for the ids list instead
    # of generating it ourselves.

    # Here we sanatize the distances list by changing all ids into indexes in
    # the ids list and making sure all distances are positive.
    dists = [[ids.index(d[0]), ids.index(d[1]), abs(d[2])] for d in distances]
    positions = initial_positions

    # use gradient descent for the given number of iterations
    for i in range(iterations):
        positions, energy = descend(positions, dists, rate)
        print(i, energy)

    return [[ids[i], p] for i, p in zip(count(), positions)]
