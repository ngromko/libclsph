
/**
 * @brief      Locates the particle data for a certain grid cell in the cell
 *table.
 *
 * @param[in]  cell_index       The index of the grid cell we are examining.
 * @param[in]  cell_table       A flattened representation of the grid contents
 * @param[in]  params           Contains the simulation parameters
 *
 * @return     The start and finish indexes of the subarray that contains the
 *particles that can be found at cell_index.
 *
 */
__device__ uint2 get_start_end_indices_for_cell(uint cell_index,
                                                const unsigned int* cell_table,
                                                simulation_parameters params) {
  uint2 indices = {
      cell_table[cell_index], (params.grid_cell_count > cell_index + 1)
                                  ? cell_table[cell_index + 1]
                                  : params.particles_count,
  };

  return indices;
}

/**
 * @brief Updates each particle with its position in the grid and fills an array
 *with the number of particles contained in each grid cell
 *
 * @param[in]  particles        Contains all the particle data
 * @param[out] out_particles    Will contain the particle data with the added
 *information
 * @param[in]  params           Contains the simulation parameters
 */
__global__ void locate_in_grid(const particle* particles,
                               particle* out_particles,
                               simulation_parameters params) {
  const size_t current_particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  out_particles[current_particle_index] = particles[current_particle_index];

  uint3 position_in_grid = {0, 0, 0};

  float x_min = params.min_point.x;
  float y_min = params.min_point.y;
  float z_min = params.min_point.z;

  // Grid cells will always have a radius length h
  position_in_grid.x = (uint)(
      (particles[current_particle_index].position.x - x_min) / (params.h * 2));
  position_in_grid.y = (uint)(
      (particles[current_particle_index].position.y - y_min) / (params.h * 2));
  position_in_grid.z = (uint)(
      (particles[current_particle_index].position.z - z_min) / (params.h * 2));

  uint grid_index = get_grid_index_z_curve(
      position_in_grid.x, position_in_grid.y, position_in_grid.z);

  out_particles[current_particle_index].grid_index = grid_index;
}
