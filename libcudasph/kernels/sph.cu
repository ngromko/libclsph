inline __device__ float operator*(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator*(float b, float3 a) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __device__ float3 operator+(float b, float3 a) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}

/*inline __device__ float3 operator-(float b, float3 a){
  return make_float3(a.x-b,a.y-b,a.z-b);
}*/

inline __device__ float length(float3 a) { return norm3df(a.x, a.y, a.z); }

inline __device__ float distance(float3 a, float3 b) {
  return norm3df(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float clamp(float x, float a, float b) {
  return fmaxf(a, fminf(b, x));
}
#define KERNEL_INCLUDE

extern __shared__ int local_data[];

__global__ void fillUintArray(uint* bob, uint value, uint length) {
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < length) bob[id] = value;
}

#define CUDART_PI_F 3.141592654f
__device__ float poly_6(float r, float h, precomputed_kernel_values terms) {
  return (1.f - clamp(floor(r / h), 0.f, 1.f)) * terms.poly_6 *
         powf((powf(h, 2) - powf(r, 2)), 3);
}

__device__ inline float3 poly_6_gradient(float3 r, float h,
                                         precomputed_kernel_values terms) {
  return (1.f - clamp(floor(length(r) / h), 0.f, 1.f)) * terms.poly_6_gradient *
         r * powf((powf(h, 2) - powf(length(r), 2)), 2);
}

__device__ inline float poly_6_laplacian(float r, float h,
                                         precomputed_kernel_values terms) {
  return (1.f - clamp(floor(r / h), 0.f, 1.f)) * terms.poly_6_laplacian *
         (powf(h, 2) - powf(r, 2)) * (3.f * powf(h, 2) - 7.f * powf(r, 2));
}

#define EPSILON 0.0000001f

__device__ inline float3 spiky_gradient(float3 r, float h,
                                        precomputed_kernel_values terms) {
  if (length(r) - EPSILON < 0.f && 0.f < length(r) + EPSILON) {
    float tmp = (-45.f / ((CUDART_PI_F * powf(h, 6))));
    return make_float3(tmp, tmp, tmp);
  }
  return (1.f - clamp(floor(length(r) / h), 0.f, 1.f)) * terms.spiky *
         (r / length(r)) * powf(h - length(r), 2);
}

__device__ inline float viscosity_laplacian(float r, float h,
                                            precomputed_kernel_values terms) {
  return (1.f - clamp(floor(r / h), 0.f, 1.f)) * terms.viscosity * (h - r);
}

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

__device__ float compute_density_with_grid(
    size_t current_particle_index, const particle* others,
    const simulation_parameters params,
    const precomputed_kernel_values smoothing_terms,
    const unsigned int* grid_cell_particle_list) {
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

/*float3 compute_internal_forces_with_grid(
    size_t current_particle_index,  const particle* others,
    const simulation_parameters params,
    const precomputed_kernel_values smoothing_terms,
    const unsigned int* grid_cell_particle_list) {
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
                (others[i].pressure / powf(others[i].density, 2) +
                 others[current_particle_index].pressure /
                     powf(others[current_particle_index].density, 2)) *
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
}*/

typedef struct {
  float3 position, next_velocity;
  int collision_happened;
  float time_elapsed;
  int indice;
} collision_response;

__device__ int respond(collision_response* response, float3 p, float3 normal,
                       float restitution, float d, float time_elapsed) {
  response->position = p + d * normal;

  response->next_velocity =
      response->next_velocity -
      (1.f +
       restitution * d / (time_elapsed * length(response->next_velocity))) *
          dot(response->next_velocity, normal) * normal;
  return 1;
}

__device__ float det(float x1, float y1, float x2, float y2) {
  return x1 * y2 - y1 * x2;
}

__device__ float distPointDroite(float x, float y, float z, float x1, float y1,
                                 float x2, float y2) {
  float A = y - x1;
  float B = z - y1;
  float C = x2 - x1;
  float D = y2 - y1;

  float dot = A * C + B * D;
  float len_sq = C * C + D * D;
  float param = -1;
  if (len_sq != 0)  // in case of 0 length line
    param = dot / len_sq;

  float xx, yy;

  if (param < 0) {
    xx = x1;
    yy = y1;
  } else if (param > 1) {
    xx = x2;
    yy = y2;
  } else {
    xx = x1 + param * C;
    yy = y1 + param * D;
  }

  float dy = y - xx;
  float dz = z - yy;
  return sqrt(x * x + dz * dz + dy * dy);
}

__global__ void kernelComputeDistanceField(float* df, const BB* bboxs,
                                           const float* transforms,
                                           const float* rvertices,
                                           uint face_count, uint gridcount) {
  int indice = face_count - 1;
  int toffset = bboxs[indice].offset;
  float temd = 20;
  const unsigned int current_df_index = blockIdx.x * blockDim.x + threadIdx.x;
  while (toffset > current_df_index && indice > 0) {
    indice--;
    toffset = bboxs[indice].offset;
  }
  if (current_df_index < gridcount) {
    int x = ((current_df_index - toffset) %
             (bboxs[indice].size_x * bboxs[indice].size_z)) %
            bboxs[indice].size_x;
    int z = ((current_df_index - toffset) %
             (bboxs[indice].size_x * bboxs[indice].size_z)) /
            bboxs[indice].size_x;
    int y = (current_df_index - toffset) /
            (bboxs[indice].size_x * bboxs[indice].size_z);

    float px = x * (bboxs[indice].maxx - bboxs[indice].minx) /
                   (bboxs[indice].size_x - 1) +
               bboxs[indice].minx;
    float py = y * (bboxs[indice].maxy - bboxs[indice].miny) /
                   (bboxs[indice].size_y - 1) +
               bboxs[indice].miny;
    float pz = z * (bboxs[indice].maxz - bboxs[indice].minz) /
                   (bboxs[indice].size_z - 1) +
               bboxs[indice].minz;

    for (int i = 0; i < face_count; i++) {
      if (px <= bboxs[i].maxx && px >= bboxs[i].minx && py <= bboxs[i].maxy &&
          py >= bboxs[i].miny && pz <= bboxs[i].maxz && pz >= bboxs[i].minz) {
        float tpx = px + transforms[i * 12 + 3];
        float tpy = py + transforms[i * 12 + 7];
        float tpz = pz + transforms[i * 12 + 11];

        float rpx = transforms[i * 12] * tpx + transforms[i * 12 + 1] * tpy +
                    transforms[i * 12 + 2] * tpz;
        float rpy = transforms[i * 12 + 4] * tpx +
                    transforms[i * 12 + 5] * tpy + transforms[i * 12 + 6] * tpz;
        float rpz = transforms[i * 12 + 8] * tpx +
                    transforms[i * 12 + 9] * tpy +
                    transforms[i * 12 + 10] * tpz;

        float v1y = rvertices[4 * i + 1];
        float v2x = rvertices[4 * i + 2];
        float v2y = rvertices[4 * i + 3];

        float a = det(rpy, rpz, 0, v1y) / det(v2x, v2y, 0, v1y);

        float b = -det(rpy, rpz, v2x, v2y) / det(v2x, v2y, 0, v1y);

        float d, td;
        if (a > 0 && b > 0 && a + b < 1)
          d = fabs(rpx);
        else {
          d = distPointDroite(rpx, rpy, rpz, 0, 0, rvertices[4 * i],
                              rvertices[4 * i + 1]);
          td = distPointDroite(rpx, rpy, rpz, rvertices[4 * i],
                               rvertices[4 * i + 1], rvertices[4 * i + 2],
                               rvertices[4 * i + 3]);
          if (td < d) {
            d = td;
          }
          td = distPointDroite(rpx, rpy, rpz, 0, 0, rvertices[4 * i + 2],
                               rvertices[4 * i + 3]);
          if (td < d) {
            d = td;
          }
        }
        if (d < fabs(temd)) {
          temd = copysignf(d, rpx);
        }
      }
    }

    df[current_df_index] = temd;
  }
}

__device__ float weigthedAverage(float x, float x1, float x2, float d1,
                                 float d2) {
  return ((x2 - x) / (x2 - x1)) * d1 + ((x - x1) / (x2 - x1)) * d2;
}

__device__ float bilinearInterpolation(float x, float y, float xmin, float ymin,
                                       float xmax, float ymax, float d00,
                                       float d01, float d10, float d11) {
  float R1 = weigthedAverage(x, xmin, xmax, d00, d10);
  float R2 = weigthedAverage(x, xmin, xmax, d01, d11);
  return weigthedAverage(y, ymin, ymax, R1, R2);
}

__device__ int getDFindex(BB bbox, float x, float y, float z, short a, short b,
                          short c) {
  return bbox.offset + (y + b) * bbox.size_x * bbox.size_z +
         bbox.size_x * (z + c) + x + a;
}

__device__ collision_response
handle_collisions(float3 old_position, float3 position, float3 next,
                  float restitution, float time_elapsed, const float* df,
                  const BB* bboxs, uint face_count) {
  int indice = -1;
  collision_response response = {position, next, 0, time_elapsed, -1};

  for (int i = 0; i < face_count; i++) {
    if (position.x <= bboxs[i].maxx && position.x >= bboxs[i].minx &&
        position.y <= bboxs[i].maxy && position.y >= bboxs[i].miny &&
        position.z <= bboxs[i].maxz && position.z >= bboxs[i].minz) {
      indice = i;
    }
  }

  if (indice > -1) {
    float sidex =
        (bboxs[indice].maxx - bboxs[indice].minx) / (bboxs[indice].size_x - 1);
    float sidey =
        (bboxs[indice].maxy - bboxs[indice].miny) / (bboxs[indice].size_y - 1);
    float sidez =
        (bboxs[indice].maxz - bboxs[indice].minz) / (bboxs[indice].size_z - 1);

    int x = (position.x - bboxs[indice].minx) / (sidex);
    int y = (position.y - bboxs[indice].miny) / (sidey);
    int z = (position.z - bboxs[indice].minz) / (sidez);

    float bx = x * sidex + bboxs[indice].minx;
    float by = y * sidey + bboxs[indice].miny;
    float bz = z * sidez + bboxs[indice].minz;

    float facedown = bilinearInterpolation(
        position.x, position.z, bx, bz, bx + sidex, bz + sidez,
        df[getDFindex(bboxs[indice], x, y, z, 0, 0, 0)],
        df[getDFindex(bboxs[indice], x, y, z, 0, 0, 1)],
        df[getDFindex(bboxs[indice], x, y, z, 1, 0, 0)],
        df[getDFindex(bboxs[indice], x, y, z, 1, 0, 1)]);
    float faceup = bilinearInterpolation(
        position.x, position.z, bx, bz, bx + sidex, bz + sidez,
        df[getDFindex(bboxs[indice], x, y, z, 0, 1, 0)],
        df[getDFindex(bboxs[indice], x, y, z, 0, 1, 1)],
        df[getDFindex(bboxs[indice], x, y, z, 1, 1, 0)],
        df[getDFindex(bboxs[indice], x, y, z, 1, 1, 1)]);

    float d = weigthedAverage(position.y, by, by + sidey, facedown, faceup);
    response.indice = indice;
    if (d < 0.02) {
      response.collision_happened = 1;
      response.indice = indice * 10;
      float faceright = bilinearInterpolation(
          position.y, position.z, by, bz, by + sidey, bz + sidez,
          df[getDFindex(bboxs[indice], x, y, z, 1, 0, 0)],
          df[getDFindex(bboxs[indice], x, y, z, 1, 0, 1)],
          df[getDFindex(bboxs[indice], x, y, z, 1, 1, 0)],
          df[getDFindex(bboxs[indice], x, y, z, 1, 1, 1)]);
      float faceleft = bilinearInterpolation(
          position.y, position.z, by, bz, by + sidey, bz + sidez,
          df[getDFindex(bboxs[indice], x, y, z, 0, 0, 0)],
          df[getDFindex(bboxs[indice], x, y, z, 0, 0, 1)],
          df[getDFindex(bboxs[indice], x, y, z, 0, 1, 0)],
          df[getDFindex(bboxs[indice], x, y, z, 0, 1, 1)]);

      float faceback = bilinearInterpolation(
          position.x, position.y, bx, by, bx + sidex, by + sidey,
          df[getDFindex(bboxs[indice], x, y, z, 0, 0, 0)],
          df[getDFindex(bboxs[indice], x, y, z, 0, 1, 0)],
          df[getDFindex(bboxs[indice], x, y, z, 1, 0, 0)],
          df[getDFindex(bboxs[indice], x, y, z, 1, 1, 0)]);
      float facefront = bilinearInterpolation(
          position.x, position.y, bx, by, bx + sidex, by + sidey,
          df[getDFindex(bboxs[indice], x, y, z, 0, 0, 1)],
          df[getDFindex(bboxs[indice], x, y, z, 0, 1, 1)],
          df[getDFindex(bboxs[indice], x, y, z, 1, 0, 1)],
          df[getDFindex(bboxs[indice], x, y, z, 1, 1, 1)]);

      float3 normal = {(faceright - faceleft), (faceup - facedown),
                       (facefront - faceback)};
      float lenn = length(normal);
      normal = normal / lenn;

      respond(&response, position, normal, restitution, fabs(d), time_elapsed);
      response.time_elapsed =
          time_elapsed * (length(response.position - old_position) /
                          length(position - old_position));
    }
  }

  return response;
}

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

typedef struct {
  float3 old_position, new_position, next_velocity;
} advection_result;

__device__ advection_result advect(float3 current_position,
                                   float3 intermediate_velocity,
                                   float3 acceleration, float time_elapsed) {
  advection_result res;

  res.old_position = current_position;

  // Leapfrog
  res.next_velocity = intermediate_velocity + acceleration * time_elapsed;

  res.new_position = current_position + res.next_velocity * time_elapsed;

  return res;
}

__global__ void density_pressure(
    const particle* input_data, particle* output_data,
    const simulation_parameters params,
    const precomputed_kernel_values smoothing_terms,
    const unsigned int* cell_table) {
  /* we'll get the same amount of global_ids as there are particles */
  const size_t current_particle_index = blockIdx.x * blockDim.x + threadIdx.x;

  particle current_particle = input_data[current_particle_index];

  current_particle.density = compute_density_with_grid(
      current_particle_index, input_data, params, smoothing_terms, cell_table);

  output_data[current_particle_index] = current_particle;

  // Tait equation more suitable to liquids than state equation
  output_data[current_particle_index].pressure =
      params.K *
      (powf(current_particle.density / params.fluid_density, 7) - 1.f);
}

__global__ void forces(const particle* input_data, particle* output_data,
                       const simulation_parameters params,
                       const precomputed_kernel_values smoothing_terms,
                       const unsigned int* cell_table) {
  const size_t current_particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t local_index = threadIdx.x;

  particle* mlocal_Data = (particle*)local_data;
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
        uint2 indices =
            get_start_end_indices_for_cell(grid_index, cell_table, params);

        for (size_t i = indices.x; i < indices.y; ++i) {
          if (blockDim.x * blockIdx.x <= i &&
              i < blockDim.x * (blockIdx.x + 1)) {
            other = mlocal_Data[i - blockDim.x * blockIdx.x];
          } else {
            other = input_data[i];
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

            viscosity_term =
                viscosity_term +
                ((other.velocity - mlocal_Data[local_index].velocity) *
                 (params.particle_mass / other.density) *
                 viscosity_laplacian(
                     length(mlocal_Data[local_index].position - other.position),
                     params.h, smoothing_terms));
          }

          normal = normal + (params.particle_mass / other.density *
                             poly_6_gradient(mlocal_Data[local_index].position -
                                                 other.position,
                                             params.h, smoothing_terms));

          color_field_laplacian =
              color_field_laplacian +
              (params.particle_mass / other.density *
               poly_6_laplacian(
                   length(mlocal_Data[local_index].position - other.position),
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
      sum / mlocal_Data[local_index].density;

  output_data[current_particle_index].acceleration =
      output_data[current_particle_index].acceleration +
      params.constant_acceleration;

  // Copy back the information into the ouput buffer
  // output_data[current_particle_index] = output_particle;
}

__global__ void advection_collision(
    const particle* input_data, particle* output_data, const float restitution,
    const float time_delta, const precomputed_kernel_values smoothing_terms,
    const unsigned int* cell_table, const float* df, const BB* bboxs,
    uint face_count) {
  const size_t current_particle_index = blockIdx.x * blockDim.x + threadIdx.x;
  output_data[current_particle_index] = input_data[current_particle_index];
  particle output_particle = input_data[current_particle_index];

  float time_to_go = time_delta;
  collision_response response;
  float3 current_position = input_data[current_particle_index].position;
  float3 current_velocity =
      input_data[current_particle_index].intermediate_velocity;
  float3 acceleration = output_particle.acceleration;

  // do {
  advection_result res =
      advect(current_position, current_velocity, acceleration, time_to_go);

  response =
      handle_collisions(res.old_position, res.new_position, res.next_velocity,
                        restitution, time_to_go, df, bboxs, face_count);

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
