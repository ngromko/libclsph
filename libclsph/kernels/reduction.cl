
__kernel
void minimum_pos(__global particle* buffer,
    __local float3* scratch,
    __const int length,
    __global float3* result) {

    int global_index = get_global_id(0);
    float3 accumulator = {INFINITY,INFINITY,INFINITY};
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float3 element = buffer[global_index].position;
        accumulator.x = min(accumulator.x,element.x);
        accumulator.y = min(accumulator.y,element.y);
        accumulator.z = min(accumulator.z,element.z);
        global_index += get_global_size(0);
    }

  // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2;
        offset > 0;
        offset = offset / 2) {
        if (local_index < offset) {
            float3 other = scratch[local_index + offset];
            float3 mine = scratch[local_index];
            scratch[local_index].x = min(mine.x,other.x);
            scratch[local_index].y = min(mine.y,other.y);
            scratch[local_index].z = min(mine.z,other.z);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}


void kernel maximum_pos(global particle* buffer,
    local float3* scratch,
    const int length,
    global float3* result) {

    int global_index = get_global_id(0);
    float3 accumulator = {-INFINITY,-INFINITY,-INFINITY};
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float3 element = buffer[global_index].position;
        accumulator.x = max(accumulator.x,element.x);
        accumulator.y = max(accumulator.y,element.y);
        accumulator.z = max(accumulator.z,element.z);
        global_index += get_global_size(0);
    }

  // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2;
        offset > 0;
        offset = offset / 2) {
        if (local_index < offset) {
            float3 other = scratch[local_index + offset];
            float3 mine = scratch[local_index];
            scratch[local_index].x = max(mine.x,other.x);
            scratch[local_index].y = max(mine.y,other.y);
            scratch[local_index].z = max(mine.z,other.z);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}

__kernel
void maximum_vit(__global particle* buffer,
    __local float* scratch,
    __const int length,
    __global float* result) {

    int global_index = get_global_id(0);
    float accumulator = -INFINITY;
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float3 element = buffer[global_index].velocity;
        float vit = element.x*element.x + element.y*element.y + element.z*element.z;
        accumulator = max(accumulator,vit);
        global_index += get_global_size(0);
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2;
        offset > 0;
        offset = offset / 2) {
        if (local_index < offset) {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = max(mine,other);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}

__kernel
void maximum_accel(__global particle* buffer,
    __local float* scratch,
    __const int length,
    __global float* result) {

    int global_index = get_global_id(0);
    float accumulator = -INFINITY;
    // Loop sequentially over chunks of input vector
    while (global_index < length) {
        float3 element = buffer[global_index].acceleration;
        float vit = element.x*element.x + element.y*element.y + element.z*element.z;
        accumulator = max(accumulator,vit);
        global_index += get_global_size(0);
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int offset = get_local_size(0) / 2;
        offset > 0;
        offset = offset / 2) {
        if (local_index < offset) {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = max(mine,other);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}

