

void kernel density_pressure(global const particle* input_data,
                             //__local particle* local_data,
                             global particle* output_data,
                             const simulation_parameters params,
                             const precomputed_kernel_values smoothing_terms,
                             global const unsigned int* cell_table) {
  /* we'll get the same amount of global_ids as there are particles */
  const size_t current_particle_index = get_global_id(0);
  const size_t group_index = get_group_id(0);
  const size_t index_in_group = get_local_id(0);
  const size_t group_size = get_local_size(0);

  /* First let's copy the data we'll use to local memory
  event_t e = async_work_group_copy(
      (__local char*)local_data,
      (__global const char*)input_data +
          (group_index * group_size * (sizeof(particle) / sizeof(char))),
      group_size * (sizeof(particle) / sizeof(char)), 0);
  wait_group_events(1, &e);

  particle current_particle = local_data[index_in_group];*/
  particle current_particle = input_data[current_particle_index];

  current_particle.density = compute_density_with_grid(
      current_particle_index, input_data, params, smoothing_terms, cell_table);

  output_data[current_particle_index] = current_particle;

  // Tait equation more suitable to liquids than state equation
  output_data[current_particle_index].pressure =
      params.K *
      (pown(current_particle.density / params.fluid_density, 7) - 1.f);
}

void kernel forces(global const particle* input_data,
                   global particle* output_data,
                   const simulation_parameters params,
                   const precomputed_kernel_values smoothing_terms,
                   global const unsigned int* cell_table) {
  const size_t current_particle_index = get_global_id(0);

  particle output_particle;

  output_data[current_particle_index] = input_data[current_particle_index];

  output_particle.acceleration =
      compute_internal_forces_with_grid(current_particle_index, input_data,
                                        params, smoothing_terms, cell_table) /
      input_data[current_particle_index].density;

  output_particle.acceleration += params.constant_acceleration;

  // Copy back the information into the ouput buffer
  output_data[current_particle_index] = output_particle;
}

void kernel advection_collision(global const particle* input_data,
                                global particle* output_data,
                                const simulation_parameters params,
                                const precomputed_kernel_values smoothing_terms,
                                global const unsigned int* cell_table,
                                global const float* face_normals,
                                global const float* vertices,
                                global const uint* indices, uint face_count) {
  const size_t current_particle_index = get_global_id(0);
  output_data[current_particle_index] = input_data[current_particle_index];
  particle output_particle = input_data[current_particle_index];

  float time_to_go = params.time_delta * params.simulation_scale;
  collision_response response;
  float3 current_position = input_data[current_particle_index].position;
  float3 current_velocity =
      input_data[current_particle_index].intermediate_velocity;

  do {
    advection_result res =
        advect(current_position, current_velocity, output_particle.acceleration,
               params.max_velocity, time_to_go);

    response =
        handle_collisions(res.old_position, res.new_position, res.next_velocity,
                          params.restitution, time_to_go, face_normals,
                          vertices, indices, face_count);

    current_position = response.position;
    current_velocity = response.next_velocity;

    time_to_go -= response.time_elapsed;

    output_particle.acceleration.x = 0.f;
    output_particle.acceleration.y = 0.f;
    output_particle.acceleration.z = 0.f;

  } while (response.collision_happened);

  output_particle.velocity =
      (input_data[current_particle_index].intermediate_velocity +
       response.next_velocity) /
      2.f;
  output_particle.intermediate_velocity = response.next_velocity;
  output_particle.position = response.position;

  // Copy back the information into the ouput buffer
  output_data[current_particle_index] = output_particle;
}

