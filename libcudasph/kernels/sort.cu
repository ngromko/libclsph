__constant__ const unsigned int mask = 0xFF;

__device__ inline unsigned int get_count_offset(int index, unsigned int mask,
                                                int pass_number,
                                                int radix_width) {
  return (index & (mask << (pass_number * radix_width))) >>
         (pass_number * radix_width);
}

__device__ inline uint2 get_start_and_end(size_t particle_count,
                                          int thread_count, int work_item_id) {
  size_t particles_per_thread = particle_count / thread_count;
  size_t start_index = particles_per_thread * work_item_id;
  size_t end_index = start_index + particles_per_thread - 1;

  if (work_item_id == thread_count - 1) {
    end_index = particle_count - 1;
  }

  return make_uint2(start_index, end_index);
}

/* size of counts = sizeof(size_t) * bucket_count * thread_count */
__global__ void sort_count(const particle* particles,
                           volatile unsigned int* counts,
                           simulation_parameters params, int thread_count,
                           int pass_number, int radix_width) {
  size_t work_item_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint2 indices =
      get_start_and_end(params.particles_count, thread_count, work_item_id);

  for (size_t i = indices.x; i <= indices.y; ++i) {
    unsigned int bucket = get_count_offset(particles[i].grid_index, mask,
                                           pass_number, radix_width);

    size_t counts_index = bucket * thread_count + work_item_id;

    ++(counts[counts_index]);
  }
}

__global__ void sort(const particle* in_particles, particle* out_particles,
                     uint* start_indices, simulation_parameters params,
                     int thread_count, int pass_number, int radix_width) {
  const size_t work_item_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint2 indices =
      get_start_and_end(params.particles_count, thread_count, work_item_id);

  for (size_t i = indices.x; i <= indices.y; ++i) {
    unsigned int bucket = get_count_offset(in_particles[i].grid_index, mask,
                                           pass_number, radix_width);

    size_t insertion_index =
        start_indices[bucket * thread_count + work_item_id];
    ++(start_indices[bucket * thread_count + work_item_id]);

    out_particles[insertion_index] = in_particles[i];
  }
}

__global__ void fill_cell_table(const particle* particles, uint* cell_table,
                                uint particle_count, uint cell_count) {
  const size_t work_item_id = blockIdx.x * blockDim.x + threadIdx.x;

  uint current_index = particles[work_item_id].grid_index;

  if (work_item_id <= particles[0].grid_index) {
    cell_table[work_item_id] = 0;
  }
  if (work_item_id > 0) {
    uint diff = current_index - particles[work_item_id - 1].grid_index;
    for (uint i = 0; i < diff; i++) {
      cell_table[current_index] = work_item_id;
      current_index--;
    }
  }
}
