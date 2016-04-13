#define _USE_MATH_DEFINES
#include <cmath>
#include <thread>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#define EXIT_ON_CL_ERROR

#include "util/pico_json/picojson.h"
//#include "util/cereal/archives/binary.hpp"

#define KERNEL_INCLUDE
#include "sph_simulation.cuh"
#include "common/util.cuh"
#include "kernels/sph.cu"
#include "cuda_profiler_api.h"

#define check_cuda_error(error) if(error != cudaSuccess){ std::cerr << "A CUDA error occured (" << __FILE__ << ":" << __LINE__ << ")-> " <<error<< cudaGetErrorString(error) << std::endl; exit(2);}


const size_t kPreferredWorkGroupSizeMultiple = 32;

/**
 * @brief Restores the state of the last simulation or places the particles in
 *the shape of a cube if no state is found.
 *
 * @param[out] buffer        Points to the particles buffer to be filled
 * @param[in] parameters    Contains the simulation parameters
 *
 */
void sph_simulation::init_particles(particle* buffer,
                                    const simulation_parameters& parameters) {
  int particles_per_cube_side = ceil(cbrtf(parameters.particles_count));
  float side_length = cbrtf(initial_volume);
  float spacing = side_length / (float)particles_per_cube_side;

  std::cout << "volume: " << initial_volume << " side_length: " << side_length
            << " spacing: " << spacing << std::endl;

  // Last simualtion serialized its last frame
  // Lets load that and pick up where it let off
  /*std::filebuf fb;
  if (fb.open("last_frame.bin", std::ios::in)) {
    std::istream file_in(&fb);

    cereal::BinaryInputArchive archive(file_in);
    archive.loadBinary(buffer, sizeof(particle) * parameters.particles_count);

    fb.close();
  }*/
  // No serialized particle array was found
  // Initialize the particles in their default position
  //else {
    for (unsigned int i = 0; i < parameters.particles_count; ++i) {
      // Arrange the particles in the form of a cube
      buffer[i].position.x =
          (float)(i % particles_per_cube_side) * spacing - side_length / 2.f;
      buffer[i].position.y =
          ((float)((i / particles_per_cube_side) % particles_per_cube_side) *
           spacing);
      buffer[i].position.z =
          (float)(i / (particles_per_cube_side * particles_per_cube_side)) *
              spacing -
          side_length / 2.f;

      buffer[i].velocity.x = 0.f;
      buffer[i].velocity.y = 0.f;
      buffer[i].velocity.z = 0.f;
      buffer[i].intermediate_velocity.x = 0.f;
      buffer[i].intermediate_velocity.y = 0.f;
      buffer[i].intermediate_velocity.z = 0.f;

      buffer[i].density = 0.f;
      buffer[i].pressure = 0.f;
    }
  }
//}

/**
 * @brief Sorts the particles according to their grid index using Radix Sort
 *
 * @param[in] first_buffer       The first OpenCL buffer used
 * @param[in] second_buffer      The second OpenCL buffer used
 * @param[out] cell_table_buffer The OpenCl buffer that contains the start indexes of
 *the cell in the sorted array
 *
 */
void sph_simulation::sort_particles(particle* first_buffer,
                                    particle* second_buffer,
                                    unsigned int* cell_table_buffer) {

  for (int pass_number = 0; pass_number < 4; ++pass_number) {

    fillUintArray<<<kBucketCount,kSortThreadCount>>>(
      sort_count_buffer_,0,kSortThreadCount * kBucketCount);

        sort_count<<<1,kSortThreadCount>>>(first_buffer,sort_count_buffer_,parameters,kSortThreadCount,pass_number,kRadixWidth);

    cudaMemcpy(sort_count_array_.data(),
        sort_count_buffer_, sizeof(unsigned int) * kSortThreadCount * kBucketCount, cudaMemcpyDeviceToHost
        );

    unsigned int running_count = 0;
    for (int i = 0; i < kSortThreadCount * kBucketCount; ++i) {
      unsigned int tmp = sort_count_array_[i];
      sort_count_array_[i] = running_count;
      running_count += tmp;
    }

   cudaMemcpy(sort_count_buffer_,sort_count_array_.data(), sizeof(unsigned int) * kSortThreadCount * kBucketCount, cudaMemcpyHostToDevice
        );

    sort<<<1,kSortThreadCount>>>(first_buffer, second_buffer,
                    sort_count_buffer_, parameters, kSortThreadCount,
                    pass_number, kRadixWidth);

    particle* tmp = first_buffer;
    first_buffer = second_buffer;
    second_buffer = tmp;
  }


         fillUintArray<<<getNumBlock(parameters.grid_cell_count),size_of_groups>>>(cell_table_buffer,parameters.particles_count,parameters.grid_cell_count);

  // Build the cell table by computing the cumulative sum at every cell.

          fill_cell_table<<<kParticlesBlocks,size_of_groups>>>(first_buffer,cell_table_buffer,parameters.particles_count,parameters.grid_cell_count);

}

float sph_simulation::simulate_single_frame(particle* front_buffer,
                                           particle* back_buffer,
                                           float dt) {

  findMinMaxPosition(front_buffer);

  // Locate each particle in the grid and build the grid count table
  locate_in_grid<<<kParticlesBlocks,size_of_groups>>>(front_buffer, back_buffer,
                  parameters);

  unsigned int* cell_table_buffer;
  cudaMalloc((void**)&cell_table_buffer,parameters.grid_cell_count*sizeof(unsigned int));

  sort_particles(back_buffer, front_buffer, cell_table_buffer);

  // Compute the density and the pressure term at every particle.
  density_pressure<<<kParticlesBlocks,size_of_groups>>>(back_buffer,
    front_buffer, parameters, precomputed_terms, cell_table_buffer);

  // Compute the density-forces at every particle.
 forces<<<kParticlesBlocks,size_of_groups,size_of_groups*sizeof(particle)>>>(
    front_buffer, back_buffer, parameters, precomputed_terms, cell_table_buffer);

  // Advect particles and resolve collisions with scene geometry.
  float cumputedTime = dt;
  do{
      dt= cumputedTime;
      advection_collision<<<kParticlesBlocks,size_of_groups>>>(
        back_buffer, front_buffer, parameters.restitution,dt, precomputed_terms, cell_table_buffer,
        df_buffer_, bb_buffer_, current_scene.face_count);

      cumputedTime= computeTimeStep(front_buffer);
  }while(dt-cumputedTime>0.00001);

  cudaFree(cell_table_buffer);
  return cumputedTime;
}

void sph_simulation::simulate() {

  std::thread savet;

  bool readParticle = true;
cudaProfilerStart();
  particle* front_buffer, *back_buffer;

  cudaMalloc((void**)&front_buffer, sizeof(particle)*parameters.particles_count);
  cudaMalloc((void**)&back_buffer, sizeof(particle)*parameters.particles_count);

  cudaMalloc((void**)&df_buffer_, sizeof(float)*current_scene.totalGridpoints);
  cudaMalloc((void**)&bb_buffer_, sizeof(BB)*current_scene.bbs.size());

  cudaMemcpy(bb_buffer_,current_scene.bbs.data(),sizeof(BB)*current_scene.bbs.size(),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&sort_count_buffer_, sizeof(unsigned int)*kSortThreadCount * kBucketCount);

  particle* particles = new particle[parameters.particles_count];
  init_particles(particles, parameters);

  //-----------------------------------------------------
  // Initial transfer to the GPU
  //-----------------------------------------------------
  cudaMemcpy(front_buffer,particles,sizeof(particle)*parameters.particles_count,cudaMemcpyHostToDevice);

  //Get number of groups for reduction
  cudaDeviceGetAttribute(&size_of_groups,cudaDevAttrMaxThreadsPerBlock,0);
  max_size_of_groups = size_of_groups;
  cudaDeviceGetAttribute(&max_unit,cudaDevAttrMultiProcessorCount,0);
  int maxLocalMem;
  cudaDeviceGetAttribute(&maxLocalMem, cudaDevAttrMaxSharedMemoryPerBlock,0);
    // Calculate the optimal size for workgroups

  // Start groups size at their maximum, make them smaller if necessary
  // Optimally parameters.particles_count should be devisible by
  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  // Refer to CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
  while((parameters.particles_count%size_of_groups!=0) || (size_of_groups*sizeof(particle) > maxLocalMem))
  {
    size_of_groups /= 2;
  }

  kParticlesBlocks = parameters.particles_count/size_of_groups;

  //Make sure that the workgroups are small enough and that the particle data will fit in local memory
  assert(size_of_groups <= max_size_of_groups );
  assert(size_of_groups*sizeof(particle) <= maxLocalMem);

  float time = 0.0;
  float timeperframe = 1.0f/parameters.target_fps;
  int currentFrame = 2;
  float dt = timeperframe*parameters.simulation_scale;

  computeDistanceField();

  if(save_frame){
    savet = std::thread([=] { save_frame(particles,parameters); });
  }

  while(time<parameters.simulation_time)
  {
    std::cout << "Simulating frame " << currentFrame << " (" << time<< "s)" << std::endl;

    if (!write_intermediate_frames && pre_frame) {
        readParticle = executePreFrameOpperation(particles,front_buffer,readParticle);
    }

    float timeleft=timeperframe;
    while(timeleft > 0.0) {
      if (write_intermediate_frames && pre_frame){
         readParticle = executePreFrameOpperation(particles,front_buffer,readParticle);
      }
      readParticle=true;
      dt=simulate_single_frame(front_buffer,back_buffer,dt);
      check_cuda_error(cudaGetLastError());
      timeleft-=dt;
      if(timeleft<dt){
          dt=timeleft;
      }
      std::cout<<"temps restant pour la frame :"<<timeleft<<std::endl;
      if(save_frame && write_intermediate_frames){
        cudaMemcpy(particles,front_buffer,sizeof(particle)*parameters.particles_count,cudaMemcpyDeviceToHost);
        readParticle = false;
        savet.join();

        savet = std::thread([=] { save_frame(particles,parameters); });
      }
      if (write_intermediate_frames && post_frame){
          readParticle = executePostFrameOpperation(particles,front_buffer,readParticle);
      }
    }
    time+=timeperframe;

    ++currentFrame;

    if(!write_intermediate_frames && save_frame){
      cudaMemcpy(particles,front_buffer,sizeof(particle)*parameters.particles_count,cudaMemcpyDeviceToHost);
      readParticle = false;
      savet.join();
      savet = std::thread([=] { save_frame(particles,parameters); });
    }

    if (!write_intermediate_frames && post_frame) {
      readParticle = executePostFrameOpperation(particles,front_buffer,readParticle);
    }
  }
  savet.join();
  cudaFree(df_buffer_);
  cudaFree(bb_buffer_);
  cudaFree(front_buffer);
  cudaFree(back_buffer);
  cudaFree(sort_count_buffer_);
  cudaProfilerStop();
  delete[] particles;
}

void sph_simulation::load_settings(std::string fluid_file_name,
                                   std::string parameters_file_name) {
  int particles_inside_influence_radius = 0;

  {
    picojson::value fluid_params;
    std::ifstream fluid_stream(fluid_file_name);

    std::string err = picojson::parse(fluid_params, fluid_stream);
    if (! err.empty()) {
      std::cerr << err << std::endl;
    }

    if (! fluid_params.is<picojson::object>()) {
      std::cerr << fluid_file_name << " is not an JSON object" << std::endl;
      exit(2);
    }

    fluid_params.get<picojson::object>()["fluid_density"];
    parameters.fluid_density =
        (float)(fluid_params.get<picojson::object>()["fluid_density"]
                    .get<double>());
    parameters.dynamic_viscosity =
        (float)(fluid_params.get<picojson::object>()["dynamic_viscosity"]
                    .get<double>());
    parameters.restitution =
        (float)(fluid_params.get<picojson::object>()["restitution"]
                    .get<double>());
    if (parameters.restitution < 0 || parameters.restitution > 1) {
      throw std::runtime_error("Restitution has an invalid value!");
    }

    parameters.K =
        (float)(fluid_params.get<picojson::object>()["k"].get<double>());
    parameters.surface_tension_threshold =
        (float)(fluid_params
                    .get<picojson::object>()["surface_tension_threshold"]
                    .get<double>());
    parameters.surface_tension =
        (float)(fluid_params.get<picojson::object>()["surface_tension"]
                    .get<double>());
    particles_inside_influence_radius =
        (int)(fluid_params
                  .get<picojson::object>()["particles_inside_influence_radius"]
                  .get<double>());
  }

  {
    picojson::value sim_params;
    std::ifstream sim_stream(parameters_file_name);

    std::string err = picojson::parse(sim_params, sim_stream);
    if (! err.empty()) {
      std::cerr << err << std::endl;
    }

    if (! sim_params.is<picojson::object>()) {
      std::cerr << parameters_file_name << " is not an JSON object" << std::endl;
      exit(2);
    }

    parameters.particles_count =
        (unsigned int)(sim_params.get<picojson::object>()["particles_count"]
                           .get<double>());

    if (parameters.particles_count % kPreferredWorkGroupSizeMultiple != 0) {
      std::cout << std::endl
                << "\033[1;31m You should choose a number of particles that is "
                   "divisble by the preferred work group size.\033[0m";
      std::cout << std::endl
                << "\033[1;31m Performances will be sub-optimal.\033[0m"
                << std::endl;
    }

    parameters.particle_mass =
        (float)(sim_params.get<picojson::object>()["particle_mass"]
                    .get<double>());
    parameters.simulation_time =
        (float)(sim_params.get<picojson::object>()["simulation_time"]
                    .get<double>());
    parameters.target_fps =
        (float)(sim_params.get<picojson::object>()["target_fps"].get<double>());
    parameters.simulation_scale =
        (float)(sim_params.get<picojson::object>()["simulation_scale"].get<double>());

    parameters.constant_acceleration.x =
        (float)(sim_params.get<picojson::object>()["constant_acceleration"]
                    .get<picojson::object>()["x"]
                    .get<double>());
    parameters.constant_acceleration.y =
        (float)(sim_params.get<picojson::object>()["constant_acceleration"]
                    .get<picojson::object>()["y"]
                    .get<double>());
    parameters.constant_acceleration.z =
        (float)(sim_params.get<picojson::object>()["constant_acceleration"]
                    .get<picojson::object>()["z"]
                    .get<double>());

    write_intermediate_frames =
        sim_params.get<picojson::object>()["write_all_frames"].get<bool>();
    serialize = sim_params.get<picojson::object>()["serialize"].get<bool>();
  }

  parameters.total_mass = parameters.particles_count * parameters.particle_mass;
  initial_volume = parameters.total_mass / parameters.fluid_density;
  parameters.h = cbrtf(3.f * (particles_inside_influence_radius *
                              (initial_volume / parameters.particles_count)) /
                       (4.f * M_PI));

  precomputed_terms.poly_6 = 315.f / (64.f * M_PI * pow(parameters.h, 9.f));
  precomputed_terms.poly_6_gradient =
      -945.f / (32.f * M_PI * pow(parameters.h, 9.f));
  precomputed_terms.poly_6_laplacian =
      -945.f / (32.f * M_PI * pow(parameters.h, 9.f));
  precomputed_terms.spiky = -45.f / (M_PI * pow(parameters.h, 6.f));
  precomputed_terms.viscosity = 45.f / (M_PI * pow(parameters.h, 6.f));
}

//--------------------------------------------------------------------------------------------------
// computeTimeStep
float sph_simulation::computeTimeStep(particle* input_buffer)
{
    // Find maximum velocity and acceleration
    float redresult[max_unit];
    float* reducResult;

    cudaMalloc((void**)&reducResult, sizeof(float) * max_unit);

    //-----------------------------------------------------
    // Find maximum speed
    //-----------------------------------------------------
    maximum_vit<<<max_unit,max_size_of_groups,max_size_of_groups*sizeof(float)>>>(
      input_buffer, parameters.particles_count, reducResult);

    cudaMemcpy(
        redresult,reducResult, sizeof(float) * max_unit, cudaMemcpyDeviceToHost);


    for(size_t i = 1; i < max_unit; ++i) {
        if(redresult[i] > redresult[0]) redresult[0]= redresult[i];
    }

    float maxVel2 = redresult[0];
    float maxVel = sqrt(redresult[0]);

    //-----------------------------------------------------
    // Find maximum acceleration
    //-----------------------------------------------------
    maximum_accel<<<max_unit,max_size_of_groups,max_size_of_groups*sizeof(float)>>>(
      input_buffer, parameters.particles_count, reducResult);

    cudaMemcpy(
        redresult,reducResult, sizeof(float) * max_unit, cudaMemcpyDeviceToHost);

    for(size_t i = 1; i < max_unit; ++i) {
        if(redresult[i] > redresult[0]) redresult[0]= redresult[i];
    }

    float maxAccel = sqrt(redresult[0]);

    //if(maxVel>15)
    //std::cin.ignore();

    // Compute timestep
    //float speedOfSound = sqrt(parameters.K);
    //float tf = sqrt(parameters.h / maxAccel);
    //float tcv = parameters.h / (speedOfSound + 0.6*(speedOfSound + 2.0/_maxuij));
    //float _dt = (tf < tcv) ? tf : tcv;
    float dt = (sqrt(2*maxAccel*parameters.h+maxVel2)-maxVel)/(2*maxAccel);
    // Clamp time step
    if (dt < 0.00001) dt = 0.00001;
    if (dt > 1.0f/(parameters.target_fps)*parameters.simulation_scale) dt = 1.0f/(parameters.target_fps)*parameters.simulation_scale;

    cudaFree(reducResult);
    return dt;
}

void sph_simulation::computeDistanceField(){

  float* trans;
  cudaMalloc((void**)&trans, sizeof(float) * current_scene.transforms.size());

  float* rvert;
  cudaMalloc((void**)&rvert, sizeof(float) * current_scene.rvertices.size());

  cudaMemcpy(trans,current_scene.transforms.data(), sizeof(float) * current_scene.transforms.size(),cudaMemcpyHostToDevice);

  cudaMemcpy(rvert,current_scene.rvertices.data(), sizeof(float) * current_scene.rvertices.size(),cudaMemcpyHostToDevice);

  kernelComputeDistanceField<<<getNumBlock(current_scene.totalGridpoints),size_of_groups>>>(
    df_buffer_, bb_buffer_, trans, rvert, current_scene.face_count, current_scene.totalGridpoints);

  cudaFree(trans);
  cudaFree(rvert);
}

void sph_simulation::findMinMaxPosition(particle* input_buffer){

    float3 redresult[max_unit];
    float3* reducResult;
    float grid_cell_side_length = (parameters.h * 2);

    cudaMalloc((void**)&reducResult, sizeof(float3) * max_unit);

    //-----------------------------------------------------
    // Find minimum postion
    //-----------------------------------------------------
    minimum_pos<<<max_unit,max_size_of_groups,max_size_of_groups*sizeof(float3)>>>(
      input_buffer, parameters.particles_count, reducResult);

    cudaMemcpy(
        redresult,reducResult, sizeof(float3) * max_unit, cudaMemcpyDeviceToHost);

    for(size_t i = 1; i < max_unit; ++i) {

        if(redresult[i].x < redresult[0].x) redresult[0].x = redresult[i].x;
        if(redresult[i].y < redresult[0].y) redresult[0].y = redresult[i].y;
        if(redresult[i].z < redresult[0].z) redresult[0].z = redresult[i].z;
    }

    // Subtracts a cell length to all sides to create a padding layer
    // This simplifies calculations further down the line
    parameters.min_point.x = redresult[0].x - 2*grid_cell_side_length;
    parameters.min_point.y = redresult[0].y - 2*grid_cell_side_length;
    parameters.min_point.z = redresult[0].z - 2*grid_cell_side_length;

    maximum_pos<<<max_unit,max_size_of_groups,max_size_of_groups*sizeof(float3)>>>(
      input_buffer, parameters.particles_count, reducResult);

    cudaMemcpy(
        redresult,reducResult, sizeof(float3) * max_unit, cudaMemcpyDeviceToHost);

    for(size_t i = 1; i < max_unit; ++i) {
        if(redresult[i].x > redresult[0].x) redresult[0].x = redresult[i].x;
        if(redresult[i].y > redresult[0].y) redresult[0].y = redresult[i].y;
        if(redresult[i].z > redresult[0].z) redresult[0].z = redresult[i].z;
    }

    // Adds a cell length to all sides to create a padding layer
    // This simplifies calculations further down the line
    parameters.max_point.x = redresult[0].x + 2*grid_cell_side_length;
    parameters.max_point.y = redresult[0].y + 2*grid_cell_side_length;
    parameters.max_point.z = redresult[0].z + 2*grid_cell_side_length;

    //-----------------------------------------------------
    // Recalculate the boundaries of the grid since the particles probably moved
    // since the last frame.
    //-----------------------------------------------------

    parameters.grid_size_x = static_cast<unsigned int>((parameters.max_point.x - parameters.min_point.x) / (grid_cell_side_length));
    parameters.grid_size_y = static_cast<unsigned int>((parameters.max_point.y - parameters.min_point.y) / (grid_cell_side_length));
    parameters.grid_size_z = static_cast<unsigned int>((parameters.max_point.z - parameters.min_point.z) / (grid_cell_side_length));


    // The Z-curve uses interleaving of bits in a uint to caculate the index.
    // This means we have floor(32/dimension_count) bits to represent each
    // dimension.
    assert(parameters.grid_size_x < 1024);
    assert(parameters.grid_size_y < 1024);
    assert(parameters.grid_size_z < 1024);

    parameters.grid_cell_count = get_grid_index_z_curve(
        parameters.grid_size_x,
        parameters.grid_size_y,
        parameters.grid_size_z
    );
    cudaFree(reducResult);
}

bool sph_simulation::executePreFrameOpperation(particle* particles, particle* buffer, bool readParticle){

    //Only read particle if not already done
    if(readParticle){
        cudaMemcpy(particles,buffer,sizeof(particle)*parameters.particles_count,cudaMemcpyDeviceToHost);
        readParticle=false;
    }
    //Only write particle id needed
  if(pre_frame(particles, parameters)){
      cudaMemcpy(buffer,particles,sizeof(particle)*parameters.particles_count,cudaMemcpyHostToDevice);
  }
  return readParticle;
}

bool sph_simulation::executePostFrameOpperation(particle* particles, particle* buffer, bool readParticle){

    //Only read particle if not already done
    if(readParticle){
        cudaMemcpy(particles,buffer,sizeof(particle)*parameters.particles_count,cudaMemcpyDeviceToHost);
        readParticle=false;
    }
    //Only write particle id needed
  if(post_frame(particles, parameters)){
      cudaMemcpy(buffer,particles,sizeof(particle)*parameters.particles_count,cudaMemcpyHostToDevice);
  }
  return readParticle;
}

int sph_simulation::getNumBlock(unsigned int numThreads){
  if(numThreads%kPreferredWorkGroupSizeMultiple==0)
      return numThreads/size_of_groups;
  else
      return numThreads/size_of_groups +1;
}
