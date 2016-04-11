

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
                   local particle* local_data,
                   const simulation_parameters params,
                   const precomputed_kernel_values smoothing_terms,
                   global const unsigned int* cell_table) {
  const size_t current_particle_index = get_global_id(0);
  const size_t local_index = get_local_id(0);
  const size_t group_index = get_group_id(0);
  const size_t group_size = get_local_size(0);

  local_data[local_index] = input_data[current_particle_index];
  barrier(CLK_LOCAL_MEM_FENCE);
  output_data[current_particle_index] = local_data[local_index];

  particle other;

  float3 pressure_term = {0.f, 0.f, 0.f};
  float3 viscosity_term = {0.f, 0.f, 0.f};
  // compute the inward surface normal, it's the gradient of the color field
  float3 normal = {0.f, 0.f, 0.f};
  // also need the color field laplacian
  float color_field_laplacian = 0.f;

  uint3 cell_coords =
      get_cell_coords_z_curve(local_data[local_index].grid_index);

  for (uint z = cell_coords.z - 1; z <= cell_coords.z + 1; ++z) {
    for (uint y = cell_coords.y - 1; y <= cell_coords.y + 1; ++y) {
      for (uint x = cell_coords.x - 1; x <= cell_coords.x + 1; ++x) {
        uint grid_index = get_grid_index_z_curve(x, y, z);
        uint2 indices = get_start_end_indices_for_cell(
            grid_index, cell_table, params);

        for (size_t i = indices.x; i < indices.y; ++i) {
            if(group_size*group_index <=i && i < group_size*group_index+group_size){
                other=local_data[i-group_size*group_index];
            }else{
                other=input_data[i];
            }
          if (i != current_particle_index) {
            //[kelager] (4.11)
            pressure_term +=
                (other.pressure / pown(other.density, 2) +
                 local_data[local_index].pressure /
                     pown(local_data[local_index].density, 2)) *
                params.particle_mass *
                spiky_gradient(local_data[local_index].position -
                                   other.position,
                               params.h, smoothing_terms);

            viscosity_term +=
                (other.velocity - local_data[local_index].velocity) *
                (params.particle_mass / other.density) *
                viscosity_laplacian(
                    length(local_data[local_index].position -
                           other.position),
                    params.h, smoothing_terms);
          }

          normal += params.particle_mass / other.density *
                    poly_6_gradient(local_data[local_index].position -
                                        other.position,
                                    params.h, smoothing_terms);

          color_field_laplacian +=
              params.particle_mass / other.density *
              poly_6_laplacian(length(local_data[local_index].position -
                                      other.position),
                               params.h, smoothing_terms);
        }
      }
    }
  }

  float3 sum = (-local_data[local_index].density * pressure_term) +
               (viscosity_term * params.dynamic_viscosity);

  if (length(normal) > params.surface_tension_threshold) {
    sum += -params.surface_tension * color_field_laplacian * normal /
           length(normal);
  }

  output_data[current_particle_index].acceleration =
      sum /local_data[local_index].density;

  output_data[current_particle_index].acceleration += params.constant_acceleration;

  // Copy back the information into the ouput buffer
  //output_data[current_particle_index] = output_particle;
}

void kernel advection_collision_const(global const particle* input_data,
                                global particle* output_data,
                                const float restitution,
                                const float time_delta,
                                const precomputed_kernel_values smoothing_terms,
                                global const unsigned int* cell_table,
                                global const float* df,
                                constant const BB* bboxs,
                                uint face_count
                                ) {
  const size_t current_particle_index = get_global_id(0);
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
        handle_collisions_const(
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

void kernel advection_collision(global const particle* input_data,
                                global particle* output_data,
                                const float restitution,
                                const float time_delta,
                                const precomputed_kernel_values smoothing_terms,
                                global const unsigned int* cell_table,
                                global const float* df,
                                global const BB* bboxs,
                                uint face_count
                                ) {
  const size_t current_particle_index = get_global_id(0);
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

