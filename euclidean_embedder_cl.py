import numpy as np
import pyopencl as cl
from itertools import count
import os


def embed(ids, distances, initial_positions, rate, iterations):
    # Here we sanatize the distances list by changing all ids into indexes in
    # the ids list and making sure all distances are positive.
    pair_indices = np.asarray([[ids.index(d[0]), ids.index(d[1])] for d in distances], dtype=np.int32)
    target_dists = np.asarray([[abs(float(d[2]))] for d in distances], dtype=np.float32)
    positions = np.asarray(initial_positions, dtype=np.float32)
    position_deltas = np.zeros_like(positions, dtype=np.float32)

    # load and set up the kernal and stuff
    platform = cl.get_platforms()[0]

    # I have 2 GPU's, the second one returned is the more powerful for me.
    gpu = platform.get_devices(cl.device_type.GPU)[1]
    # cpu = platform.get_devices(cl.device_type.CPU)[0]

    context = cl.Context([gpu])

    dims = len(initial_positions[0])

    basepath = os.path.dirname(__file__)
    source_path = os.path.abspath(os.path.join(basepath, 'euclidean_descent.cl'))
    with open(source_path, 'r') as source:
        code = source.read().replace('SIZE', str(dims))
        code = code.replace('RATE', str(rate))

    program = cl.Program(context, code).build()
    queue = cl.CommandQueue(context)

    print("Creating buffers")
    positions_buf = cl.Buffer(context,
                              cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=positions)

    pair_indices_buf = cl.Buffer(context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=pair_indices)

    target_dists_buf = cl.Buffer(context,
                                 cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=target_dists)

    position_delta_buf = cl.Buffer(context,
                                   cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=position_deltas)

    # use gradient descent for the given number of iterations
    print("Iterating")
    for i in range(iterations):

        program.descend(queue, (len(target_dists),), None,
                        positions_buf, pair_indices_buf,
                        target_dists_buf, position_delta_buf)

        program.update(queue, (len(positions),), None,
                       positions_buf, position_delta_buf)

    cl.enqueue_copy(queue, positions, positions_buf)
    return [[ids[i]] + list(p) for i, p in zip(count(), positions)]
