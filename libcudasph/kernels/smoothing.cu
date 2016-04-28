
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
