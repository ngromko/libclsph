inline __device__ float operator*(float3 a, float3 b){
  return a.x*b.x+a.y*b.y+a.z*b.z;
}

inline __device__ float dot(float3 a, float3 b){
  return a.x*b.x+a.y*b.y+a.z*b.z;
}

inline __device__ float3 operator*(float3 a, float b){
  return make_float3(a.x*b,a.y*b,a.z*b);
}

inline __device__ float3 operator*(float b, float3 a){
  return make_float3(a.x*b,a.y*b,a.z*b);
}

inline __device__ float3 operator/(float3 a, float b){
  return make_float3(a.x/b,a.y/b,a.z/b);
}

inline __device__ float3 operator+(float3 a, float3 b){
  return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);
}

inline __device__ float3 operator+(float3 a, float b){
  return make_float3(a.x+b,a.y+b,a.z+b);
}

inline __device__ float3 operator+(float b, float3 a){
  return make_float3(a.x+b,a.y+b,a.z+b);
}

inline __device__ float3 operator-(float3 a, float3 b){
  return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);
}

inline __device__ float3 operator-(float3 a, float b){
  return make_float3(a.x-b,a.y-b,a.z-b);
}

/*inline __device__ float3 operator-(float b, float3 a){
  return make_float3(a.x-b,a.y-b,a.z-b);
}*/

inline __device__ float length(float3 a){
    return norm3df(a.x,a.y,a.z);
}

inline __device__ float distance(float3 a, float3 b){
    return norm3df(a.x-b.x,a.y-b.y,a.z-b.z);
}

inline __device__ float clamp(float x, float a, float b)
{
  return fmaxf(a, fminf(b, x));
}
