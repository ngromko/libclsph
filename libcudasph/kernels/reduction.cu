
__global__ void minimum_pos(particle* buffer, __const int length,
                            float3* result) {
  float3* scratch = (float3*)local_data;
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  float3 accumulator = {INFINITY, INFINITY, INFINITY};
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float3 element = buffer[global_index].position;
    accumulator.x = fminf(accumulator.x, element.x);
    accumulator.y = fminf(accumulator.y, element.y);
    accumulator.z = fminf(accumulator.z, element.z);
    global_index += gridDim.x * blockDim.x;
  }

  // Perform parallel reduction
  int local_index = threadIdx.x;
  scratch[local_index] = accumulator;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      float3 other = scratch[local_index + offset];
      float3 mine = scratch[local_index];
      scratch[local_index].x = fminf(mine.x, other.x);
      scratch[local_index].y = fminf(mine.y, other.y);
      scratch[local_index].z = fminf(mine.z, other.z);
    }
    __syncthreads();
  }
  if (local_index == 0) {
    result[blockIdx.x] = scratch[0];
  }
}

__global__ void maximum_pos(particle* buffer, const int length,
                            float3* result) {
  float3* scratch = (float3*)local_data;
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  float3 accumulator = {-INFINITY, -INFINITY, -INFINITY};
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float3 element = buffer[global_index].position;
    accumulator.x = fmaxf(accumulator.x, element.x);
    accumulator.y = fmaxf(accumulator.y, element.y);
    accumulator.z = fmaxf(accumulator.z, element.z);
    global_index += gridDim.x * blockDim.x;
  }

  // Perform parallel reduction
  int local_index = threadIdx.x;
  scratch[local_index] = accumulator;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      float3 other = scratch[local_index + offset];
      float3 mine = scratch[local_index];
      scratch[local_index].x = fmaxf(mine.x, other.x);
      scratch[local_index].y = fmaxf(mine.y, other.y);
      scratch[local_index].z = fmaxf(mine.z, other.z);
    }
    __syncthreads();
  }
  if (local_index == 0) {
    result[blockIdx.x] = scratch[0];
  }
}

__global__ void maximum_vit(particle* buffer, int length, float* result) {
  float* scratch = (float*)local_data;
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  float accumulator = -INFINITY;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float3 element = buffer[global_index].velocity;
    float vit =
        element.x * element.x + element.y * element.y + element.z * element.z;
    accumulator = fmaxf(accumulator, vit);
    global_index += gridDim.x * blockDim.x;
  }

  // Perform parallel reduction
  int local_index = threadIdx.x;
  scratch[local_index] = accumulator;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = fmaxf(mine, other);
    }
    __syncthreads();
  }
  if (local_index == 0) {
    result[blockIdx.x] = scratch[0];
  }
}

__global__ void maximum_accel(particle* buffer, int length, float* result) {
  float* scratch = (float*)local_data;
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  float accumulator = -INFINITY;
  // Loop sequentially over chunks of input vector
  while (global_index < length) {
    float3 element = buffer[global_index].acceleration;
    float vit =
        element.x * element.x + element.y * element.y + element.z * element.z;
    accumulator = fmaxf(accumulator, vit);
    global_index += gridDim.x * blockDim.x;
  }

  // Perform parallel reduction
  int local_index = threadIdx.x;
  scratch[local_index] = accumulator;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
    if (local_index < offset) {
      float other = scratch[local_index + offset];
      float mine = scratch[local_index];
      scratch[local_index] = fmaxf(mine, other);
    }
    __syncthreads();
  }
  if (local_index == 0) {
    result[blockIdx.x] = scratch[0];
  }
}
