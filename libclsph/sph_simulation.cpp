#define _USE_MATH_DEFINES
#include <cmath>

#define EXIT_ON_CL_ERROR

#include "util/pico_json/picojson.h"
#include "util/cereal/archives/binary.hpp"

#include "sph_simulation.h"
#include "common/util.h"

const size_t kPreferredWorkGroupSizeMultiple = 64;

const std::string BUFFER_KERNEL_FILE_NAME = "kernels/sph.cl";

cl::Device* running_device;

cl::Kernel make_kernel(cl::Program& p, const char* name) {

  cl_int cl_error;
  cl::Kernel k = cl::Kernel(p, name, &cl_error);
  check_cl_error(cl_error);

  return k;
}

void set_kernel_args_internal(int, cl::Kernel&) { return; }

template <typename T1, typename... TArgs>
void set_kernel_args_internal(int index, cl::Kernel& kernel, T1 a,
                              TArgs... args) {
  check_cl_error(kernel.setArg(index, a));
  set_kernel_args_internal(index + 1, kernel, args...);
}

template <typename... TArgs>
void set_kernel_args(cl::Kernel& kernel, TArgs... args) {
  set_kernel_args_internal(0, kernel, args...);
}

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
  std::filebuf fb;
  if (fb.open("last_frame.bin", std::ios::in)) {
    std::istream file_in(&fb);

    cereal::BinaryInputArchive archive(file_in);
    archive.loadBinary(buffer, sizeof(particle) * parameters.particles_count);

    fb.close();
  }
  // No serialized particle array was found
  // Initialize the particles in their default position
  else {
    for (unsigned int i = 0; i < parameters.particles_count; ++i) {
      // Arrange the particles in the form of a cube
      buffer[i].position.s[0] =
          (float)(i % particles_per_cube_side) * spacing - side_length / 2.f;
      buffer[i].position.s[1] =
          ((float)((i / particles_per_cube_side) % particles_per_cube_side) *
           spacing);
      buffer[i].position.s[2] =
          (float)(i / (particles_per_cube_side * particles_per_cube_side)) *
              spacing -
          side_length / 2.f;

      buffer[i].velocity.s[0] = 0.f;
      buffer[i].velocity.s[1] = 0.f;
      buffer[i].velocity.s[2] = 0.f;
      buffer[i].intermediate_velocity.s[0] = 0.f;
      buffer[i].intermediate_velocity.s[1] = 0.f;
      buffer[i].intermediate_velocity.s[2] = 0.f;

      buffer[i].density = 0.f;
      buffer[i].pressure = 0.f;
    }
  }
}

/**
 * @brief Sorts the particles according to their grid index using Radix Sort
 *
 * @param[in] first_buffer       The first OpenCL buffer used
 * @param[in] second_buffer      The second OpenCL buffer used
 * @param[out] cell_table_buffer The OpenCl buffer that contains the start indexes of
 *the cell in the sorted array
 *
 */
void sph_simulation::sort_particles(cl::Buffer& first_buffer,
                                    cl::Buffer& second_buffer,
                                    cl::Buffer& cell_table_buffer) {
  cl::Buffer* current_input_buffer = &first_buffer;
  cl::Buffer* current_output_buffer = &second_buffer;

  for (int pass_number = 0; pass_number < 4; ++pass_number) {

    set_kernel_args(kernel_fill_uint_array_,sort_count_buffer_,0,kSortThreadCount * kBucketCount);
    check_cl_error(queue_.enqueueNDRangeKernel(
        kernel_fill_uint_array_, cl::NullRange, cl::NDRange(kSortThreadCount * kBucketCount),
        cl::NullRange));

    set_kernel_args(kernel_sort_count_, *current_input_buffer,
                    sort_count_buffer_, parameters, kSortThreadCount,
                    pass_number, kRadixWidth);

    check_cl_error(queue_.enqueueNDRangeKernel(
        kernel_sort_count_, cl::NullRange, cl::NDRange(kSortThreadCount),
        cl::NullRange));

    check_cl_error(queue_.enqueueReadBuffer(
        sort_count_buffer_, CL_TRUE, 0,
        sizeof(unsigned int) * kSortThreadCount * kBucketCount,
        sort_count_array_.data()));

    unsigned int running_count = 0;
    for (int i = 0; i < kSortThreadCount * kBucketCount; ++i) {
      unsigned int tmp = sort_count_array_[i];
      sort_count_array_[i] = running_count;
      running_count += tmp;
    }

    check_cl_error(queue_.enqueueWriteBuffer(
        sort_count_buffer_, CL_TRUE, 0,
        sizeof(unsigned int) * kSortThreadCount * kBucketCount,
        sort_count_array_.data()));

    set_kernel_args(kernel_sort_, *current_input_buffer, *current_output_buffer,
                    sort_count_buffer_, parameters, kSortThreadCount,
                    pass_number, kRadixWidth);

    check_cl_error(queue_.enqueueNDRangeKernel(kernel_sort_, cl::NullRange,
                                               cl::NDRange(kSortThreadCount),
                                               cl::NullRange));

    cl::Buffer* tmp = current_input_buffer;
    current_input_buffer = current_output_buffer;
    current_output_buffer = tmp;
  }

  set_kernel_args(kernel_fill_uint_array_,cell_table_buffer,parameters.particles_count,parameters.grid_cell_count);
  check_cl_error(
          queue_.enqueueNDRangeKernel(
              kernel_fill_uint_array_, cl::NullRange,
              cl::NDRange(parameters.grid_cell_count), cl::NullRange));


  // Build the cell table by computing the cumulative sum at every cell.
  set_kernel_args(kernel_cell_table,*current_input_buffer, cell_table_buffer, parameters.particles_count,parameters.grid_cell_count);
      check_cl_error(
          queue_.enqueueNDRangeKernel(
              kernel_cell_table, cl::NullRange,
              cl::NDRange(parameters.particles_count), cl::NullRange));
}

float sph_simulation::simulate_single_frame(cl::Buffer& front_buffer,
                                           cl::Buffer& back_buffer,
                                           float dt) {

  findMinMaxPosition(front_buffer);

  // Locate each particle in the grid and build the grid count table
  set_kernel_args(kernel_locate_in_grid_, front_buffer, back_buffer,
                  parameters);

  check_cl_error(queue_.enqueueNDRangeKernel(
      kernel_locate_in_grid_, cl::NullRange,
      cl::NDRange(parameters.particles_count), cl::NDRange(size_of_groups)));


  cl::Buffer cell_table_buffer(
      context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      sizeof(unsigned int) * parameters.grid_cell_count);

  sort_particles(back_buffer, front_buffer, cell_table_buffer);

  // Compute the density and the pressure term at every particle.
  check_cl_error(kernel_density_pressure_.setArg(0, back_buffer));
  /*check_cl_error(kernel_density_pressure_.setArg(
      1, size_of_groups * sizeof(particle),
      nullptr));  // Declare local memory in arguments*/
  check_cl_error(kernel_density_pressure_.setArg(1, front_buffer));
  check_cl_error(kernel_density_pressure_.setArg(2, parameters));
  check_cl_error(kernel_density_pressure_.setArg(3, precomputed_terms));
  check_cl_error(kernel_density_pressure_.setArg(4, cell_table_buffer));

  check_cl_error(queue_.enqueueNDRangeKernel(
      kernel_density_pressure_, cl::NullRange,
      cl::NDRange(parameters.particles_count), cl::NDRange(size_of_groups)));

  // Compute the density-forces at every particle.
  set_kernel_args(kernel_forces_, front_buffer, back_buffer, parameters,
                  precomputed_terms, cell_table_buffer);

  check_cl_error(queue_.enqueueNDRangeKernel(
      kernel_forces_, cl::NullRange, cl::NDRange(parameters.particles_count),
      cl::NDRange(size_of_groups)));

  set_kernel_args(kernel_advection_collision_, back_buffer, front_buffer,
                  parameters.restitution,dt, precomputed_terms, cell_table_buffer,
                  df_buffer_, bb_buffer_, current_scene.face_count);

  // Advect particles and resolve collisions with scene geometry.
  float cumputedTime = dt;
  do{
      dt= cumputedTime;
      check_cl_error(kernel_advection_collision_.setArg(3, dt));
      check_cl_error(queue_.enqueueNDRangeKernel(
          kernel_advection_collision_, cl::NullRange,
          cl::NDRange(parameters.particles_count), cl::NDRange(size_of_groups)));

      cumputedTime= computeTimeStep(front_buffer);
  }while(dt-cumputedTime>0.00001);

  return cumputedTime;
}

void sph_simulation::simulate() {

  cl_int cl_error;

  std::vector<cl::Device> device_array;
  check_cl_error(init_cl_single_device(&context_, device_array, "", "", true));

  queue_ = cl::CommandQueue(context_, device_array[0], 0, &cl_error);
  check_cl_error(cl_error);

  running_device = &device_array[0];

  std::string source = readKernelFile(BUFFER_KERNEL_FILE_NAME);
  cl::Program program;
  check_cl_error(make_program(&program, context_, device_array, source, true,
                              "-I ./kernels/ -I ./common/"));

  kernel_density_pressure_ = make_kernel(program, "density_pressure");
  kernel_advection_collision_ = make_kernel(program, "advection_collision");
  kernel_forces_ = make_kernel(program, "forces");
  kernel_locate_in_grid_ = make_kernel(program, "locate_in_grid");
  kernel_sort_count_ = make_kernel(program, "sort_count");
  kernel_sort_ = make_kernel(program, "sort");
  kernel_fill_uint_array_ = make_kernel(program, "fillUintArray");
  kernel_cell_table = make_kernel(program, "fill_cell_table");
  kernel_df_ = make_kernel(program, "computeDistanceField");
  kernel_minimum_pos = make_kernel(program, "minimum_pos");
  kernel_maximum_pos = make_kernel(program, "maximum_pos");
  kernel_maximum_vit = make_kernel(program, "maximum_vit");
  kernel_maximum_accel = make_kernel(program, "maximum_accel");

  cl::Buffer front_buffer_ =
      cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                 sizeof(particle) * parameters.particles_count);
  cl::Buffer back_buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            sizeof(particle) * parameters.particles_count);

  df_buffer_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY |  CL_MEM_ALLOC_HOST_PTR,
                          sizeof(float) * current_scene.totalGridpoints);

  bb_buffer_ = cl::Buffer(context_,  CL_MEM_WRITE_ONLY |  CL_MEM_ALLOC_HOST_PTR,
                          sizeof(BB) * current_scene.bbs.size());

  check_cl_error(queue_.enqueueWriteBuffer(
      bb_buffer_, CL_TRUE, 0,
      sizeof(BB) * current_scene.bbs.size(), current_scene.bbs.data()));

  sort_count_buffer_ =
      cl::Buffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                 sizeof(unsigned int) * kSortThreadCount * kBucketCount);

  particle* particles = new particle[parameters.particles_count];
  init_particles(particles, parameters);

  //-----------------------------------------------------
  // Initial transfer to the GPU
  //-----------------------------------------------------
  check_cl_error(
      queue_.enqueueWriteBuffer(
          front_buffer_, CL_TRUE, 0,
          sizeof(particle) * parameters.particles_count, particles));

  //Get number of groups for reduction
  max_size_of_groups = size_of_groups = running_device->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  max_unit= running_device->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

    // Calculate the optimal size for workgroups

  // Start groups size at their maximum, make them smaller if necessary
  // Optimally parameters.particles_count should be devisible by
  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  // Refer to CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
  while((parameters.particles_count%size_of_groups!=0) || (size_of_groups*sizeof(particle) > running_device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()))
  {
    size_of_groups /= 2;
  }

  //Make sure that the workgroups are small enough and that the particle data will fit in local memory
  assert( size_of_groups <= running_device->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() );
  assert( size_of_groups*sizeof(particle) <= running_device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() );

  float time = 0.0;
  float timeperframe = 1.0f/parameters.target_fps;
  int currentFrame = 2;
  float dt = timeperframe*parameters.simulation_scale;

  computeDistanceField();

  while(time<parameters.simulation_time)
  {
    std::cout << "Simulating frame " << currentFrame << " (" << time<< "s)" << std::endl;
    if (pre_frame) {
      pre_frame(particles, parameters, true);
    }
    float timeleft=timeperframe;

    while(timeleft > 0.0) {
      if (pre_frame) pre_frame(particles, parameters, false);
      /*if(currentFrame&1){
        dt=simulate_single_frame(back_buffer_,front_buffer_,dt);
      }
      else{*/
        dt=simulate_single_frame(front_buffer_,back_buffer_,dt);
      //}
      timeleft-=dt;
      if(timeleft<dt){
          dt=timeleft;
      }
      std::cout<<"temps restant pour la frame :"<<timeleft<<std::endl;
      if (post_frame) post_frame(particles, parameters, false);
    }
    time+=timeperframe;

    check_cl_error(queue_.enqueueReadBuffer(
        front_buffer_, CL_TRUE, 0, sizeof(particle) * parameters.particles_count,
        particles));

    ++currentFrame;
    if (post_frame) {
      post_frame(particles, parameters, true);
    }
  }
  delete[] particles;
}

void sph_simulation::load_settings(std::string fluid_file_name,
                                   std::string parameters_file_name) {
  int particles_inside_influence_radius = 0;

  {
    picojson::value fluid_params;
    std::ifstream fluid_stream(fluid_file_name);
    fluid_stream >> fluid_params;

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
    sim_stream >> sim_params;

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

    parameters.constant_acceleration.s[0] =
        (float)(sim_params.get<picojson::object>()["constant_acceleration"]
                    .get<picojson::object>()["x"]
                    .get<double>());
    parameters.constant_acceleration.s[1] =
        (float)(sim_params.get<picojson::object>()["constant_acceleration"]
                    .get<picojson::object>()["y"]
                    .get<double>());
    parameters.constant_acceleration.s[2] =
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
float sph_simulation::computeTimeStep( cl::Buffer& input_buffer)
{
    // Find maximum velocity and acceleration
   float redresult[max_unit];
   float dt;

    cl::Buffer reducResult(
        context_,
        CL_MEM_READ_WRITE ,
        sizeof(float) * max_unit);


    //-----------------------------------------------------
    // Find minimum postion
    //-----------------------------------------------------
    check_cl_error(kernel_maximum_vit.setArg(0, input_buffer));
    check_cl_error(kernel_maximum_vit.setArg(1, max_size_of_groups * sizeof(float), NULL)); //Declare local memory in arguments
    check_cl_error(kernel_maximum_vit.setArg(2, parameters.particles_count));
    check_cl_error(kernel_maximum_vit.setArg(3, reducResult));

    check_cl_error(
            queue_.enqueueNDRangeKernel(
                kernel_maximum_vit, cl::NullRange,
                cl::NDRange(max_unit*max_size_of_groups), cl::NDRange(max_size_of_groups)));


    check_cl_error(
        queue_.enqueueReadBuffer(
        reducResult,
        CL_TRUE, 0,
        sizeof(float) * max_unit,
        redresult));
    for(size_t i = 1; i < max_unit; ++i) {
        if(redresult[i] > redresult[0]) redresult[0]= redresult[i];
        //std::cout<<"max vit :"<<redresult[i]<<std::endl;
    }

    float maxVel2 = redresult[0];
    float maxVel = sqrt(redresult[0]);

    check_cl_error(kernel_maximum_accel.setArg(0, input_buffer));
    check_cl_error(kernel_maximum_accel.setArg(1, max_size_of_groups * sizeof(float), NULL)); //Declare local memory in arguments
    check_cl_error(kernel_maximum_accel.setArg(2, parameters.particles_count));
    check_cl_error(kernel_maximum_accel.setArg(3, reducResult));

    check_cl_error(
            queue_.enqueueNDRangeKernel(
                kernel_maximum_accel, cl::NullRange,
                cl::NDRange(max_unit*max_size_of_groups), cl::NDRange(max_size_of_groups)));


    check_cl_error(
        queue_.enqueueReadBuffer(
        reducResult,
        CL_TRUE, 0,
        sizeof(float) * max_unit,
        redresult));

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
    dt = (sqrt(2*maxAccel*parameters.h+maxVel2)-maxVel)/(2*maxAccel);
    // Clamp time step
    if (dt < 0.00001) dt = 0.00001;
    if (dt > 1.0f/(parameters.target_fps)*parameters.simulation_scale) dt = 1.0f/(parameters.target_fps)*parameters.simulation_scale;

    return dt;
}

void sph_simulation::computeDistanceField(){

  cl::Buffer trans(context_,  CL_MEM_WRITE_ONLY |  CL_MEM_ALLOC_HOST_PTR ,  sizeof(float) * current_scene.transforms.size());
  cl::Buffer rvert(context_,  CL_MEM_WRITE_ONLY |  CL_MEM_ALLOC_HOST_PTR ,  sizeof(float) * current_scene.rvertices.size());

  check_cl_error(
      queue_.enqueueWriteBuffer(
          trans, CL_TRUE, 0,
          sizeof(float) * current_scene.transforms.size(), current_scene.transforms.data()));

  check_cl_error(
      queue_.enqueueWriteBuffer(
          rvert, CL_TRUE, 0,
          sizeof(float) * current_scene.rvertices.size(), current_scene.rvertices.data()));

  check_cl_error(kernel_df_.setArg(0,df_buffer_));
  check_cl_error(kernel_df_.setArg(1,bb_buffer_));
  check_cl_error(kernel_df_.setArg(2,trans));
  check_cl_error(kernel_df_.setArg(3,rvert));
  check_cl_error(kernel_df_.setArg(4,current_scene.face_count));
  check_cl_error(kernel_df_.setArg(5,current_scene.totalGridpoints));

  std::cout << "totalgpoints " << current_scene.totalGridpoints << std::endl;
  check_cl_error(
          queue_.enqueueNDRangeKernel(
              kernel_df_, cl::NullRange,
              cl::NDRange(current_scene.totalGridpoints), cl::NullRange));
}

void sph_simulation::findMinMaxPosition(cl::Buffer& input_buffer){

    cl_float3 redresult[max_unit];
    cl_float grid_cell_side_length = (parameters.h * 2);

    cl::Buffer reducResult(
        context_,
        CL_MEM_READ_WRITE ,
        sizeof(cl_float3) * max_unit);


    //-----------------------------------------------------
    // Find minimum postion
    //-----------------------------------------------------
    check_cl_error(kernel_minimum_pos.setArg(0, input_buffer));
    check_cl_error(kernel_minimum_pos.setArg(1, max_size_of_groups * sizeof(cl_float3), NULL)); //Declare local memory in arguments
    check_cl_error(kernel_minimum_pos.setArg(2, parameters.particles_count));
    check_cl_error(kernel_minimum_pos.setArg(3, reducResult));

    check_cl_error(
            queue_.enqueueNDRangeKernel(
                kernel_minimum_pos, cl::NullRange,
                cl::NDRange(max_unit*max_size_of_groups), cl::NDRange(max_size_of_groups)));

    check_cl_error(
        queue_.enqueueReadBuffer(
        reducResult,
        CL_TRUE, 0,
        sizeof(cl_float3) * max_unit,
        redresult));

    for(size_t i = 1; i < max_unit; ++i) {

        if(redresult[i].s[0] < redresult[0].s[0]) redresult[0].s[0] = redresult[i].s[0];
        if(redresult[i].s[1] < redresult[0].s[1]) redresult[0].s[1] = redresult[i].s[1];
        if(redresult[i].s[2] < redresult[0].s[2]) redresult[0].s[2] = redresult[i].s[2];
    }

    // Subtracts a cell length to all sides to create a padding layer
    // This simplifies calculations further down the line
    parameters.min_point.s[0] = redresult[0].s[0] - 2*grid_cell_side_length;
    parameters.min_point.s[1] = redresult[0].s[1] - 2*grid_cell_side_length;
    parameters.min_point.s[2] = redresult[0].s[2] - 2*grid_cell_side_length;

    check_cl_error(kernel_maximum_pos.setArg(0, input_buffer));
    check_cl_error(kernel_maximum_pos.setArg(1, max_size_of_groups * sizeof(cl_float3), NULL)); //Declare local memory in arguments
    check_cl_error(kernel_maximum_pos.setArg(2, parameters.particles_count));
    check_cl_error(kernel_maximum_pos.setArg(3, reducResult));

    check_cl_error(
            queue_.enqueueNDRangeKernel(
                kernel_maximum_pos, cl::NullRange,
                cl::NDRange(max_unit*max_size_of_groups), cl::NDRange(max_size_of_groups)));

    check_cl_error(
        queue_.enqueueReadBuffer(
        reducResult,
        CL_TRUE, 0,
        sizeof(cl_float3) * max_unit,
        redresult));

    for(size_t i = 1; i < max_unit; ++i) {
        if(redresult[i].s[0] > redresult[0].s[0]) redresult[0].s[0] = redresult[i].s[0];
        if(redresult[i].s[1] > redresult[0].s[1]) redresult[0].s[1] = redresult[i].s[1];
        if(redresult[i].s[2] > redresult[0].s[2]) redresult[0].s[2] = redresult[i].s[2];
    }

    // Adds a cell length to all sides to create a padding layer
    // This simplifies calculations further down the line
    parameters.max_point.s[0] = redresult[0].s[0] + 2*grid_cell_side_length;
    parameters.max_point.s[1] = redresult[0].s[1] + 2*grid_cell_side_length;
    parameters.max_point.s[2] = redresult[0].s[2] + 2*grid_cell_side_length;

    //-----------------------------------------------------
    // Recalculate the boundaries of the grid since the particles probably moved
    // since the last frame.
    //-----------------------------------------------------

    parameters.grid_size_x = static_cast<cl_uint>((parameters.max_point.s[0] - parameters.min_point.s[0]) / (grid_cell_side_length));
    parameters.grid_size_y = static_cast<cl_uint>((parameters.max_point.s[1] - parameters.min_point.s[1]) / (grid_cell_side_length));
    parameters.grid_size_z = static_cast<cl_uint>((parameters.max_point.s[2] - parameters.min_point.s[2]) / (grid_cell_side_length));


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
}
