#ifndef _SPH_SIMULATION_H_
#define _SPH_SIMULATION_H_

#define KERNEL_INCLUDE
#include <functional>
#include "common/structures.cuh"
#include "scene.cuh"

class sph_simulation {
 public:
  sph_simulation() : parameters(), write_intermediate_frames(false) {}

  void simulate();

  simulation_parameters parameters;
  precomputed_kernel_values precomputed_terms;

  std::function<bool(particle*, const simulation_parameters&)> pre_frame;
  std::function<void(particle*, const simulation_parameters&)> save_frame;
  std::function<bool(particle*, const simulation_parameters&)> post_frame;

  void load_settings(std::string fluid_file_name,
                     std::string parameters_file_name);

  bool write_intermediate_frames;
  bool serialize;
  float initial_volume;
  scene current_scene;

 private:
  void init_particles(particle* buffer, const simulation_parameters&);
  void sort_particles(particle*, particle*, unsigned int*);
  float simulate_single_frame(particle*,particle*,float);
  float computeTimeStep(particle*);
  void computeDistanceField();
  void findMinMaxPosition(particle*);
  bool executePreFrameOpperation(particle *, particle*, bool);
  bool executePostFrameOpperation(particle *, particle*, bool);
  int getNumBlock(unsigned int);

  float* df_buffer_;
  BB* bb_buffer_;

  int max_unit;
  int size_of_groups;
  int max_size_of_groups;
  unsigned int kParticlesBlocks;

  static const int kSortThreadCount = 128;
  static const int kBucketCount = 256;
  static const int kRadixWidth = 8;

  std::array<unsigned int, kSortThreadCount * kBucketCount> sort_count_array_;
  unsigned int* sort_count_buffer_;
};

#endif
