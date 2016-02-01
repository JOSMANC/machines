from machines import euclidean_embedder
from machines import spherical_embedder
from machines import euclidean_embedder_cl
import csv
import numpy as np
import time

# Load the ids
with open('ncaa_ids.csv') as ids_file:
    reader = csv.reader(ids_file)
    ids = [name[0] for name in reader]

# Load the distance data
with open('ncaa_dists_trans1.csv') as distance_file:
    reader = csv.reader(distance_file)
    distance_list = list(reader)

# Dimensions must be 2 for the spherical embeder, but can be any
# number for euclidean embeddder
dimensions = 1

# Here I generate random initial positions, but choosing better
# initial positions should result in quicker or better results
positions = np.random.randn(len(ids), dimensions)

print("Running OpenCL")

cl_start = time.time()
results = euclidean_embedder_cl.embed(ids, distance_list, positions,
                                      rate=.0005, iterations=10000)
cl_end = time.time()

with open('ranks_cl.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)


print("Running Python")
py_start = time.time()
results = euclidean_embedder.embed(ids, distance_list, positions,
                                   rate=.0005, iterations=100)
py_end = time.time()

with open('ranks_py.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)

print("OpenCL run time: " + str(cl_end - cl_start))
print("Python run time: " + str(py_end - py_start))
