#ifndef _STRUCTURES_H_
#define _STRUCTURES_H_
#define uint unsigned int

typedef struct {
  uint particles_count;
  float fluid_density;
  float total_mass;
  float particle_mass;
  float dynamic_viscosity;
  float simulation_time;
  float h;
  float simulation_scale;
  float target_fps;
  float surface_tension_threshold;
  float surface_tension;
  float restitution;
  float K;

  float3 constant_acceleration;

  int grid_size_x;
  int grid_size_y;
  int grid_size_z;
  uint grid_cell_count;
  float3 min_point, max_point;
} simulation_parameters;

typedef struct {
  float3 position, velocity, intermediate_velocity, acceleration;
  float density, pressure;
  uint grid_index;
} particle;

typedef struct {
  float poly_6;
  float poly_6_gradient;
  float poly_6_laplacian;
  float spiky;
  float viscosity;
} precomputed_kernel_values;

typedef struct {
    float maxx;
    float maxy;
    float maxz;
    float minx;
    float miny;
    float minz;
    size_t size_x;
    size_t size_y;
    size_t size_z;
    size_t offset;
} BB;

#endif
