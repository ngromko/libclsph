#define KERNEL_INCLUDE

void kernel fillUintArray(global uint* bob ,uint value,int length){
    uint id = get_global_id(0);
    if(id<length)
        bob[id]=value;
}

#ifndef _STRUCTURES_H_
#define _STRUCTURES_H_

#ifdef KERNEL_INCLUDE
#define cl_float float
#define cl_uint unsigned int
#define cl_int int
#define cl_float3 float3
#define cl_uint3 uint3
#else
#include "util/cl_boilerplate.h"
#endif

#define COLLISION_VOLUMES_COUNT 3

typedef struct {
  cl_uint particles_count;
  cl_float max_velocity;
  cl_float fluid_density;
  cl_float total_mass;
  cl_float particle_mass;
  cl_float dynamic_viscosity;
  cl_float simulation_time;
  cl_float target_fps;
  cl_float h;
  cl_float simulation_scale;
  cl_float time_delta;
  cl_float surface_tension_threshold;
  cl_float surface_tension;
  cl_float restitution;
  cl_float K;

  cl_float3 constant_acceleration;

  cl_int grid_size_x;
  cl_int grid_size_y;
  cl_int grid_size_z;
  cl_uint grid_cell_count;
  cl_float3 min_point, max_point;
} simulation_parameters;

typedef struct {
  cl_float3 position, velocity, intermediate_velocity, acceleration;
  cl_float density, pressure;
  cl_uint grid_index;
} particle;

typedef struct {
  float poly_6;
  float poly_6_gradient;
  float poly_6_laplacian;
  float spiky;
  float viscosity;
} precomputed_kernel_values;

#endif
#ifndef _UTIL_H_
#define _UTIL_H_

cl_uint uninterleave(cl_uint value) {
  cl_uint ret = 0x0;

  ret |= (value & 0x1) >> 0;
  ret |= (value & 0x8) >> 2;
  ret |= (value & 0x40) >> 4;
  ret |= (value & 0x200) >> 6;
  ret |= (value & 0x1000) >> 8;
  ret |= (value & 0x8000) >> 10;
  ret |= (value & 0x40000) >> 12;
  ret |= (value & 0x200000) >> 14;
  ret |= (value & 0x1000000) >> 16;
  ret |= (value & 0x8000000) >> 18;

  return ret;
}

cl_uint3 get_cell_coords_z_curve(cl_uint index) {
  cl_uint mask = 0x9249249;
  cl_uint i_x = index & mask;
  cl_uint i_y = (index >> 1) & mask;
  cl_uint i_z = (index >> 2) & mask;

#ifdef KERNEL_INCLUDE
  cl_uint3 coords = {uninterleave(i_x), uninterleave(i_y), uninterleave(i_z)};
#else
  cl_uint3 coords;

  coords.s[0] = uninterleave(i_x);
  coords.s[1] = uninterleave(i_y);
  coords.s[2] = uninterleave(i_z);
#endif

  return coords;
}

// http://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
cl_uint get_grid_index_z_curve(cl_uint in_x, cl_uint in_y, cl_uint in_z) {
  cl_uint x = in_x;
  cl_uint y = in_y;
  cl_uint z = in_z;

  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;

  y = (y | (y << 16)) & 0x030000FF;
  y = (y | (y << 8)) & 0x0300F00F;
  y = (y | (y << 4)) & 0x030C30C3;
  y = (y | (y << 2)) & 0x09249249;

  z = (z | (z << 16)) & 0x030000FF;
  z = (z | (z << 8)) & 0x0300F00F;
  z = (z | (z << 4)) & 0x030C30C3;
  z = (z | (z << 2)) & 0x09249249;

  return x | (y << 1) | (z << 2);
}

#endif
inline float poly_6(float r, float h, precomputed_kernel_values terms) {
  return (1.f - clamp(floor(r / h), 0.f, 1.f)) * terms.poly_6 *
         pown((pown(h, 2) - pown(r, 2)), 3);
}

inline float3 poly_6_gradient(float3 r, float h,
                              precomputed_kernel_values terms) {
  return (1.f - clamp(floor(convert_float(length(r)) / h), 0.f, 1.f)) *
         terms.poly_6_gradient * r * pown((pown(h, 2) - pown(length(r), 2)), 2);
}

inline float poly_6_laplacian(float r, float h,
                              precomputed_kernel_values terms) {
  return (1.f - clamp(floor(convert_float(length(r)) / h), 0.f, 1.f)) *
         terms.poly_6_laplacian * (pown(h, 2) - pown(r, 2)) *
         (3.f * pown(h, 2) - 7.f * pown(r, 2));
}

#define EPSILON 0.0000001f

inline float3 spiky_gradient(float3 r, float h,
                             precomputed_kernel_values terms) {
  if (length(r) - EPSILON < 0.f && 0.f < length(r) + EPSILON) {
    return (-45.f / convert_float((M_PI * pown(h, 6))));
  }
  return (1.f - clamp(floor(convert_float(length(r)) / h), 0.f, 1.f)) *
         terms.spiky * (r / convert_float(length(r))) *
         pown(h - convert_float(length(r)), 2);
}

inline float viscosity_laplacian(float r, float h,
                                 precomputed_kernel_values terms) {
  return (1.f - clamp(floor(r / h), 0.f, 1.f)) * terms.viscosity * (h - r);
}

uint2 get_start_end_indices_for_cell(uint cell_index,
                                     global const unsigned int* cell_table,
                                     simulation_parameters params);

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
uint2 get_start_end_indices_for_cell(uint cell_index,
                                     global const unsigned int* cell_table,
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
void kernel locate_in_grid(global const particle* particles,
                           global particle* out_particles,
                           simulation_parameters params) {
  const size_t current_particle_index = get_global_id(0);
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



float compute_density_with_grid(
    size_t current_particle_index, global const particle* others,
    const simulation_parameters params,
    const precomputed_kernel_values smoothing_terms,
    global const unsigned int* grid_cell_particle_list);
float3 compute_internal_forces_with_grid(
    size_t current_particle_index, global const particle* others,
    const simulation_parameters params,
    const precomputed_kernel_values smoothing_terms,
    global const unsigned int* grid_cell_particle_list);

float compute_density_with_grid(
    size_t current_particle_index, global const particle* others,
    const simulation_parameters params,
    const precomputed_kernel_values smoothing_terms,
    global const unsigned int* grid_cell_particle_list) {
  float density = 0.f;

  uint3 cell_coords =
      get_cell_coords_z_curve(others[current_particle_index].grid_index);

  for (uint z = cell_coords.z - 1; z <= cell_coords.z + 1; ++z) {
    for (uint y = cell_coords.y - 1; y <= cell_coords.y + 1; ++y) {
      for (uint x = cell_coords.x - 1; x <= cell_coords.x + 1; ++x) {
        uint grid_index = get_grid_index_z_curve(x, y, z);
        uint2 indices = get_start_end_indices_for_cell(
            grid_index, grid_cell_particle_list, params);

        for (size_t i = indices.x; i < indices.y; ++i) {
          density += params.particle_mass *
                     poly_6(distance(others[current_particle_index].position,
                                     others[i].position),
                            params.h, smoothing_terms);
        }
      }
    }
  }

  return density;
}

float3 compute_internal_forces_with_grid(
    size_t current_particle_index, global const particle* others,
    const simulation_parameters params,
    const precomputed_kernel_values smoothing_terms,
    global const unsigned int* grid_cell_particle_list) {
  float3 pressure_term = {0.f, 0.f, 0.f};
  float3 viscosity_term = {0.f, 0.f, 0.f};
  // compute the inward surface normal, it's the gradient of the color field
  float3 normal = {0.f, 0.f, 0.f};
  // also need the color field laplacian
  float color_field_laplacian = 0.f;

  uint3 cell_coords =
      get_cell_coords_z_curve(others[current_particle_index].grid_index);

  for (uint z = cell_coords.z - 1; z <= cell_coords.z + 1; ++z) {
    for (uint y = cell_coords.y - 1; y <= cell_coords.y + 1; ++y) {
      for (uint x = cell_coords.x - 1; x <= cell_coords.x + 1; ++x) {
        uint grid_index = get_grid_index_z_curve(x, y, z);
        uint2 indices = get_start_end_indices_for_cell(
            grid_index, grid_cell_particle_list, params);

        for (size_t i = indices.x; i < indices.y; ++i) {
          if (i != current_particle_index) {
            //[kelager] (4.11)
            pressure_term +=
                (others[i].pressure / pown(others[i].density, 2) +
                 others[current_particle_index].pressure /
                     pown(others[current_particle_index].density, 2)) *
                params.particle_mass *
                spiky_gradient(others[current_particle_index].position -
                                   others[i].position,
                               params.h, smoothing_terms);

            viscosity_term +=
                (others[i].velocity - others[current_particle_index].velocity) *
                (params.particle_mass / others[i].density) *
                viscosity_laplacian(
                    length(others[current_particle_index].position -
                           others[i].position),
                    params.h, smoothing_terms);
          }

          normal += params.particle_mass / others[i].density *
                    poly_6_gradient(others[current_particle_index].position -
                                        others[i].position,
                                    params.h, smoothing_terms);

          color_field_laplacian +=
              params.particle_mass / others[i].density *
              poly_6_laplacian(length(others[current_particle_index].position -
                                      others[i].position),
                               params.h, smoothing_terms);
        }
      }
    }
  }

  float3 sum = (-others[current_particle_index].density * pressure_term) +
               (viscosity_term * params.dynamic_viscosity);

  if (length(normal) > params.surface_tension_threshold) {
    sum += -params.surface_tension * color_field_laplacian * normal /
           length(normal);
  }

  return sum;
}


typedef struct {
  float3 position, next_velocity;
  int collision_happened;
  float time_elapsed;
} collision_response;

typedef struct {
  float3 collision_point, surface_normal;
  float penetration_depth;
  int collision_happened;
} collision;

int detect_collision(collision* c, float3 p0, float3 p1,
                     global const float* face_normals,
                     global const float* vertices, global const uint* indices,
                     uint face_count) {
  c->collision_happened = 0;

  for (uint i = 0; i < face_count; ++i) {
    float3 normal = {
        face_normals[3 * i + 0], face_normals[3 * i + 1],
        face_normals[3 * i + 2],
    };

    if (dot(normal, p1 - p0) / (length(normal) * length(p1 - p0)) <= 0) {
      normal = -normal;
    }

    float3 v0 = {
        vertices[3 * indices[3 * i + 0] + 0],
        vertices[3 * indices[3 * i + 0] + 1],
        vertices[3 * indices[3 * i + 0] + 2],
    };

    float3 v1 = {
        vertices[3 * indices[3 * i + 1] + 0],
        vertices[3 * indices[3 * i + 1] + 1],
        vertices[3 * indices[3 * i + 1] + 2],
    };

    float3 v2 = {
        vertices[3 * indices[3 * i + 2] + 0],
        vertices[3 * indices[3 * i + 2] + 1],
        vertices[3 * indices[3 * i + 2] + 2],
    };

    float3 u = v1 - v0;
    float3 v = v2 - v0;

    float denom = dot(normal, p1 - p0);

    if (denom == 0.f) {
      continue;
    }

    float r = dot(normal, v0 - p0) / denom;

    if (0 <= r && r <= 1) {
      float3 intersect = p0 + r * (p1 - p0);
      float3 w = intersect - v0;
      float uv, wv, vv, wu, uu;

      uv = dot(u, v);
      wv = dot(w, v);
      vv = dot(v, v);
      wu = dot(w, u);
      uu = dot(u, u);

      float denom = uv * uv - uu * vv;
      float s = (uv * wv - vv * wu) / denom;
      float t = (uv * wu - uu * wv) / denom;

      // Collision
      if (s >= 0 && t >= 0 && s + t <= 1) {
        if (c->collision_happened &&
            length(p0 - intersect) > length(p0 - c->collision_point)) {
          continue;
        }
        c->surface_normal = normal;
        c->collision_point = intersect;
        c->penetration_depth = length(p1 - intersect);
        c->collision_happened = 1;
      }
    }
  }
  return c->collision_happened;
}

int respond(collision_response* response, collision c, float restitution,
            float time_elapsed) {
  // hack to avoid points directly on the faces, the collision detection code
  // should be
  response->position = c.collision_point - (c.surface_normal * 0.001f);

  response->next_velocity -=
      (1.f +
       restitution * c.penetration_depth /
           (time_elapsed * length(response->next_velocity))) *
      dot(response->next_velocity, c.surface_normal) * c.surface_normal;

  return 1;
}

// After the collision response, the particle's position is only partially
// updated,
// the advection  must be recursive until the entire movement is completed
collision_response handle_collisions(
    float3 old_position, float3 position, float3 next, float restitution,
    float time_elapsed, global const float* face_normals,
    global const float* vertices, global const uint* indices, uint face_count) {
  collision_response response = {
      position, next, 0, time_elapsed,
  };

  collision c;

  if (detect_collision(&c, old_position, position, face_normals, vertices,
                       indices, face_count)) {
    response.collision_happened = 1;
    respond(&response, c, restitution, time_elapsed);
    response.time_elapsed =
        time_elapsed * (length(response.position - old_position) /
                        length(position - old_position));
  }

  return response;
}
constant const unsigned int mask = 0xFF;

inline unsigned int get_count_offset(int index, unsigned int mask,
                                     int pass_number, int radix_width) {
  return (index & (mask << (pass_number * radix_width))) >>
         (pass_number * radix_width);
}

inline uint2 get_start_and_end(size_t particle_count, int thread_count,
                               int work_item_id) {
  size_t particles_per_thread = particle_count / thread_count;
  size_t start_index = particles_per_thread * work_item_id;
  size_t end_index = start_index + particles_per_thread - 1;

  if (work_item_id == thread_count - 1) {
    end_index = particle_count - 1;
  }

  return (uint2)(start_index, end_index);
}

/* size of counts = sizeof(size_t) * bucket_count * thread_count */
void kernel sort_count(global const particle* particles,
                       global volatile unsigned int* counts,
                       simulation_parameters params, int thread_count,
                       int pass_number, int radix_width) {
  size_t work_item_id = get_global_id(0);
  uint2 indices =
      get_start_and_end(params.particles_count, thread_count, work_item_id);

  for (size_t i = indices.x; i <= indices.y; ++i) {
    unsigned int bucket = get_count_offset(particles[i].grid_index, mask,
                                           pass_number, radix_width);

    size_t counts_index = bucket * thread_count + work_item_id;

    ++(counts[counts_index]);
  }
}

void kernel sort(global const particle* in_particles,
                 global particle* out_particles, global uint* start_indices,
                 simulation_parameters params, int thread_count,
                 int pass_number, int radix_width) {
  const size_t work_item_id = get_global_id(0);
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

typedef struct {
  float3 old_position, new_position, next_velocity;
} advection_result;

advection_result advect(float3 current_position, float3 intermediate_velocity,
                        float3 acceleration, float max_velocity,
                        float time_elapsed) {
  advection_result res;

  res.old_position = current_position;

  // Leapfrog
  res.next_velocity = intermediate_velocity + acceleration * time_elapsed;

  if (length(res.next_velocity) > max_velocity) {
    res.next_velocity = normalize(res.next_velocity) * max_velocity;
  }

  res.new_position = current_position + res.next_velocity * time_elapsed;

  return res;
}


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

