

__global__ void density_pressure(const particle* input_data,
                              particle* output_data,
                             const simulation_parameters params,
                             const precomputed_kernel_values smoothing_terms,
                              const unsigned int* cell_table) {
  /* we'll get the same amount of global_ids as there are particles */
  const size_t current_particle_index = blockIdx.x*blockDim.x+threadIdx.x;

  particle current_particle = input_data[current_particle_index];

  current_particle.density = compute_density_with_grid(
      current_particle_index, input_data, params, smoothing_terms, cell_table);

  output_data[current_particle_index] = current_particle;

  // Tait equation more suitable to liquids than state equation
  output_data[current_particle_index].pressure =
      params.K *
      (powf(current_particle.density / params.fluid_density, 7) - 1.f);
}

__global__ void forces(const particle* input_data,
                    particle* output_data,
                   const simulation_parameters params,
                   const precomputed_kernel_values smoothing_terms,
                  const unsigned int* cell_table) {
  const size_t current_particle_index = blockIdx.x*blockDim.x+threadIdx.x;
  const size_t local_index = threadIdx.x;

  particle* mlocal_Data = (particle*) local_data;
  mlocal_Data[local_index] = input_data[current_particle_index];

  output_data[current_particle_index] = mlocal_Data[local_index];

  particle other;

  float3 pressure_term = {0.f, 0.f, 0.f};
  float3 viscosity_term = {0.f, 0.f, 0.f};
  // compute the inward surface normal, it's the gradient of the color field
  float3 normal = {0.f, 0.f, 0.f};
  // also need the color field laplacian
  float color_field_laplacian = 0.f;

  uint3 cell_coords =
      get_cell_coords_z_curve(mlocal_Data[local_index].grid_index);

  for (uint z = cell_coords.z - 1; z <= cell_coords.z + 1; ++z) {
    for (uint y = cell_coords.y - 1; y <= cell_coords.y + 1; ++y) {
      for (uint x = cell_coords.x - 1; x <= cell_coords.x + 1; ++x) {
        uint grid_index = get_grid_index_z_curve(x, y, z);
        uint2 indices = get_start_end_indices_for_cell(
            grid_index, cell_table, params);

        for (size_t i = indices.x; i < indices.y; ++i) {
            if(blockDim.x*blockIdx.x <=i && i < blockDim.x*(blockIdx.x+1)){
                other=mlocal_Data[i-blockDim.x*blockIdx.x];
            }else{
                other=input_data[i];
            }
          if (i != current_particle_index) {
            //[kelager] (4.11)
            pressure_term = pressure_term +
                ((other.pressure / powf(other.density, 2) +
                 mlocal_Data[local_index].pressure /
                     powf(mlocal_Data[local_index].density, 2)) *
                params.particle_mass *
                spiky_gradient(mlocal_Data[local_index].position -
                                   other.position,
                               params.h, smoothing_terms));

            viscosity_term = viscosity_term+
                ((other.velocity - mlocal_Data[local_index].velocity) *
                (params.particle_mass / other.density) *
                viscosity_laplacian(
                    length(mlocal_Data[local_index].position -
                           other.position),
                    params.h, smoothing_terms));
          }

          normal = normal+ (params.particle_mass / other.density *
                    poly_6_gradient(mlocal_Data[local_index].position -
                                        other.position,
                                    params.h, smoothing_terms));

          color_field_laplacian = color_field_laplacian +
              (params.particle_mass / other.density *
              poly_6_laplacian(length(mlocal_Data[local_index].position -
                                      other.position),
                               params.h, smoothing_terms));
        }
      }
    }
  }

  float3 sum = (-mlocal_Data[local_index].density * pressure_term) +
               (viscosity_term * params.dynamic_viscosity);

  if (length(normal) > params.surface_tension_threshold) {
    sum = sum + (-params.surface_tension * color_field_laplacian * normal /
           length(normal));
  }

  output_data[current_particle_index].acceleration =
      sum /mlocal_Data[local_index].density;

  output_data[current_particle_index].acceleration = output_data[current_particle_index].acceleration + params.constant_acceleration;

  // Copy back the information into the ouput buffer
  //output_data[current_particle_index] = output_particle;
}

__global__ void advection_collision( const particle* input_data,
                                 particle* output_data,
                                const float restitution,
                                const float time_delta,
                                const precomputed_kernel_values smoothing_terms,
                                 const unsigned int* cell_table,
                                 const float* df,
                                 const BB* bboxs,
                                uint face_count
                                ) {
  const size_t current_particle_index = blockIdx.x*blockDim.x+threadIdx.x;
  output_data[current_particle_index] = input_data[current_particle_index];
  particle output_particle = input_data[current_particle_index];

  float time_to_go =time_delta;
  collision_response response;
  float3 current_position = input_data[current_particle_index].position;
  float3 current_velocity =
      input_data[current_particle_index].intermediate_velocity;
  float3 acceleration = output_particle.acceleration;

  //do {
    advection_result res =
        advect(current_position, current_velocity, acceleration,
               time_to_go);

    response =
        handle_collisions(
            res.old_position, res.new_position, res.next_velocity,
            restitution, time_to_go, df,bboxs, face_count);

    current_position = response.position;
    current_velocity = response.next_velocity;

    time_to_go -= response.time_elapsed;

    acceleration.x = 0.f;
    acceleration.y = 0.f;
    acceleration.z = 0.f;

  //} while (response.collision_happened);

  output_particle.velocity =
      (input_data[current_particle_index].intermediate_velocity +
       response.next_velocity) /
      2.f;
  output_particle.intermediate_velocity = response.next_velocity;
  output_particle.position = response.position;

  // Copy back the information into the ouput buffer
  output_data[current_particle_index] = output_particle;
}

