#define KERNEL_INCLUDE

extern __shared__ int local_data[];

__global__ void fillUintArray(uint* bob, uint value, uint length) {
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < length) bob[id] = value;
}
