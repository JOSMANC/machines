import numpy as np
from itertools import count
from math import sin, cos, acos, sqrt


def descend(radius, positions, distances, rate):
    # cumulatively store the changes in position due to each distance pair
    pos_delta = np.zeros_like(positions)
    total_energy = 0

    for distance in distances:
        # here d is target distance, s is current distance for this pair
        theta_0, phi_0 = positions[distance[0]]
        theta_1, phi_1 = positions[distance[1]]
        d = distance[2]

        # calculate the gradient of our energy: sum of (d-s)^2 over all pairs
        gamma = cos(theta_0)*cos(theta_1) + sin(theta_0)*sin(theta_1)*cos(phi_1 - phi_0)
        s = radius*acos(gamma)

        # the Jacobian must be considered for non-euclidean gradients
        t0_del = cos(theta_0)*sin(theta_1)*cos(phi_1 - phi_0) - sin(theta_0)*cos(theta_1)
        p0_del = sin(theta_1)*sin(phi_1 - phi_0)
        t1_del = sin(theta_0)*cos(theta_1)*cos(phi_1 - phi_0) - cos(theta_0)*sin(theta_1)
        p1_del = -sin(theta_0)*sin(phi_1 - phi_0)

        prefactor = (d-s) / sqrt(1 - gamma**2)
        grad_0 = prefactor*np.array([t0_del, p0_del])
        grad_1 = prefactor*np.array([t1_del, p1_del])

        # update our position deltas based on the gradient and descent rate
        pos_delta[distance[0]] -= rate*grad_0
        pos_delta[distance[1]] -= rate*grad_1
        total_energy += (d-s)**2

    # update positions and return the new positions and total energy
    return [positions + pos_delta, total_energy]


def embed(ids, distances, initial_positions, radius, rate, iterations):
    dists = [[ids.index(d[0]), ids.index(d[1]), abs(d[2])] for d in distances]
    positions = initial_positions

    # use gradient descent for the given number of iterations
    for i in range(iterations):
        positions, energy = descend(radius, positions, dists, rate)
        print(i, energy)

    return [[ids[i], p] for i, p in zip(count(), positions)]
