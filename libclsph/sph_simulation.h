#ifndef _SPH_SIMULATION_H_
#define _SPH_SIMULATION_H_

#include <functional>
#include "common/structures.h"
#include "scene.h"

class sph_simulation {
 public:
  sph_simulation() : parameters(), write_intermediate_frames(false) {}

  void simulate();

  simulation_parameters parameters;
  precomputed_kernel_values precomputed_terms;

  std::function<void(particle*, const simulation_parameters&, bool)> pre_frame;
  std::function<void(particle*, const simulation_parameters&, bool)> post_frame;

  void load_settings(std::string fluid_file_name,
                     std::string parameters_file_name);

  bool write_intermediate_frames;
  bool serialize;
  float initial_volume;
  scene current_scene;

 private:
  void init_particles(particle* buffer, const simulation_parameters&);
  void sort_particles(cl::Buffer&, cl::Buffer&, cl::Buffer&);
  float simulate_single_frame(cl::Buffer&,cl::Buffer&,float);
  float computeTimeStep(cl::Buffer&);
  void computeDistanceField();
  void findMinMaxPosition(cl::Buffer& input_buffer);

  cl::Context context_;
  cl::CommandQueue queue_;

  cl::Kernel kernel_density_pressure_;
  cl::Kernel kernel_advection_collision_;
  cl::Kernel kernel_forces_;
  cl::Kernel kernel_locate_in_grid_;
  cl::Kernel kernel_sort_count_;
  cl::Kernel kernel_sort_;
  cl::Kernel kernel_fill_uint_array_;
  cl::Kernel kernel_cell_table;
  cl::Kernel kernel_df_;
  cl::Kernel kernel_minimum_pos;
  cl::Kernel kernel_maximum_pos;
  cl::Kernel kernel_maximum_vit;
  cl::Kernel kernel_maximum_accel;

  cl::Buffer df_buffer_;
  cl::Buffer bb_buffer_;

  unsigned int max_unit;
  unsigned int size_of_groups;
  unsigned int max_size_of_groups;

  static const int kSortThreadCount = 128;
  static const int kBucketCount = 256;
  static const int kRadixWidth = 8;

  std::array<unsigned int, kSortThreadCount * kBucketCount> sort_count_array_;
  cl::Buffer sort_count_buffer_;
};

#endif
