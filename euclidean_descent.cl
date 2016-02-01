// From http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html
static void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void descend(__global float *positions,
                      __global const int2 *pair_indices,
                      __global const float *target_dists,
                      volatile global float *position_deltas){

    int pair = get_global_id(0);
    float target = target_dists[pair];
    int2 indices = pair_indices[pair];

    float dif[SIZE];
    float prefactor = 0.0;
    float s2 = 0.0;

    for(int i=0; i< SIZE; i++){
        dif[i] = positions[indices[0]*SIZE + i] - positions[indices[1]*SIZE + i];
        s2 += pow(positions[indices[0]*SIZE + i] - positions[indices[1]*SIZE + i], 2);
    }

    prefactor = (target - sqrt(s2));

    for(int i=0; i< SIZE; i++){
        atomic_add_global(&position_deltas[indices[0]*SIZE + i], RATE * prefactor * dif[i]);
        atomic_add_global(&position_deltas[indices[1]*SIZE + i], -RATE * prefactor * dif[i]);
    }
}

// Update the position values using the position_deltas, then reset
// position_deltas to 0.
__kernel void update(__global float *positions,
                     __global float *position_deltas){

    int index = get_global_id(0);
    for(int i =0; i < SIZE; i++){
        positions[index*SIZE + i] += position_deltas[index*SIZE + i];
        position_deltas[index*SIZE + i] = 0.0f;
    }
}
