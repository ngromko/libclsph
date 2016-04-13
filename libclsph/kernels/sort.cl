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

void kernel fill_cell_table(
	global const particle* particles,
	global uint* cell_table,
	uint particle_count,
	uint cell_count) {

    const size_t work_item_id = get_global_id(0);

    uint current_index=particles[work_item_id].grid_index;

    if(work_item_id<=particles[0].grid_index){
        cell_table[work_item_id]=0;
    }
    if(work_item_id>0){
        uint diff= current_index-  particles[work_item_id-1].grid_index;
        for(uint i=0;i<diff;i++){
            cell_table[current_index]=work_item_id;
            current_index--;
        }
    }
}

/* size of counts = sizeof(size_t) * bucket_count * thread_count
void kernel sort_all(global const particle* particles,
                     global volatile unsigned int* counts,
                     local uint* sharedcount,
                     uint particles_count, int groupSize,
                     uint groupNum,
                     int pass_number, int radix_width) {

    //Phase 1
  uint work_item_id = get_global_id(0);
  uint globalgroupid= work_item_id/groupSize;
  uint localgroupid = get_local_size(0)/groupSize;
  uint particles_per_thread = particle_count / thread_count;
  uint start = particles_per_thread*globalgroupid;
  uint index= start;

  while(index<particles_per_thread){
    unsigned int bucket = get_count_offset(particles[i].grid_index, mask,
                                           pass_number, radix_width);

    atomic_inc(&sharedcount[bucket*localgroupid]);
    index+=groupSize;
  }



}*/
//http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
void kernel prefix_sum_1(global const uint* in_counts,
                       global uint* out_counts,
                       local uint* temp,
                       global uint* tmpres,
                       uint n
                       ){
    int gthid = get_global_id(0);
    int thid = get_local_id(0);
    int offset = 1;

    temp[2*thid] = in_counts[2*gthid]; // load input into shared memory
    temp[2*thid+1] = in_counts[2*gthid+1];

    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        barrier(CLK_LOCAL_MEM_FENCE);
       if (thid < d)
       {
           int ai = offset*(2*thid+1)-1;
           int bi = offset*(2*thid+2)-1;
              temp[bi] += temp[ai];
       }
       offset *= 2;
    }

   if (thid == 0) {
       tmpres[get_group_id(0)]=temp[n - 1];
       temp[n - 1] = 0;
   } // clear the last element
   barrier(CLK_LOCAL_MEM_FENCE);



   for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
   {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d)
        {
           int ai = offset*(2*thid+1)-1;
           int bi = offset*(2*thid+2)-1;

           float t = temp[ai];
           temp[ai] = temp[bi];
           temp[bi] += t;
         }
   }
    barrier(CLK_LOCAL_MEM_FENCE);

   out_counts[2*gthid] = temp[2*thid]; // write results to device memory
   out_counts[2*gthid+1] = temp[2*thid+1];

}

void kernel prefix_sum_2(global const uint* blockSum_in,
                       global uint* blockSum_out,
                       local uint* temp,
                       uint n
                       ){
    int gthid = get_global_id(0);
    int thid = get_local_id(0);
    int offset = 1;

    temp[2*thid] = blockSum_in[2*gthid]; // load input into shared memory
    temp[2*thid+1] = blockSum_in[2*gthid+1];

    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        barrier(CLK_LOCAL_MEM_FENCE);
       if (thid < d)
       {
           int ai = offset*(2*thid+1)-1;
           int bi = offset*(2*thid+2)-1;
              temp[bi] += temp[ai];
       }
       offset *= 2;
    }

   if (thid == 0) {
       temp[n - 1] = 0;
   } // clear the last element
   barrier(CLK_LOCAL_MEM_FENCE);



   for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
   {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d)
        {
           int ai = offset*(2*thid+1)-1;
           int bi = offset*(2*thid+2)-1;

           float t = temp[ai];
           temp[ai] = temp[bi];
           temp[bi] += t;
         }
   }
    barrier(CLK_LOCAL_MEM_FENCE);


   blockSum_out[2*gthid] = temp[2*thid]; // write results to device memory
   blockSum_out[2*gthid+1] = temp[2*thid+1];
}

void kernel prefix_sum_3(global const uint* blockSum,
                         global uint* counts){
    int gthid = get_global_id(0);
    counts[gthid] += blockSum[gthid/128];
}
