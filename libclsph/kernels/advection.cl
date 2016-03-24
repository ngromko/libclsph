
typedef struct {
  float3 old_position, new_position, next_velocity;
} advection_result;

advection_result advect(float3 current_position, float3 intermediate_velocity,
                        float3 acceleration,
                        float time_elapsed) {
  advection_result res;

  res.old_position = current_position;

  // Leapfrog
  res.next_velocity = intermediate_velocity + acceleration * time_elapsed;

  res.new_position = current_position + res.next_velocity * time_elapsed;

  return res;
}
