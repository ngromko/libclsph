#ifndef _UTIL_H_
#define _UTIL_H_
#define uint unsigned int

__host__ __device__ uint uninterleave(uint value) {
  uint ret = 0x0;

  ret |= (value & 0x1) >> 0;
  ret |= (value & 0x8) >> 2;
  ret |= (value & 0x40) >> 4;
  ret |= (value & 0x200) >> 6;
  ret |= (value & 0x1000) >> 8;
  ret |= (value & 0x8000) >> 10;
  ret |= (value & 0x40000) >> 12;
  ret |= (value & 0x200000) >> 14;
  ret |= (value & 0x1000000) >> 16;
  ret |= (value & 0x8000000) >> 18;

  return ret;
}

__host__ __device__ uint3 get_cell_coords_z_curve(uint index) {
  uint mask = 0x9249249;
  uint i_x = index & mask;
  uint i_y = (index >> 1) & mask;
  uint i_z = (index >> 2) & mask;

#ifdef KERNEL_INCLUDE
  uint3 coords = {uninterleave(i_x), uninterleave(i_y), uninterleave(i_z)};
#else
  uint3 coords;

  coords.s[0] = uninterleave(i_x);
  coords.s[1] = uninterleave(i_y);
  coords.s[2] = uninterleave(i_z);
#endif

  return coords;
}

// http://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
__host__ __device__ uint
get_grid_index_z_curve(uint in_x, uint in_y, uint in_z) {
  uint x = in_x;
  uint y = in_y;
  uint z = in_z;

  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;

  y = (y | (y << 16)) & 0x030000FF;
  y = (y | (y << 8)) & 0x0300F00F;
  y = (y | (y << 4)) & 0x030C30C3;
  y = (y | (y << 2)) & 0x09249249;

  z = (z | (z << 16)) & 0x030000FF;
  z = (z | (z << 8)) & 0x0300F00F;
  z = (z | (z << 4)) & 0x030C30C3;
  z = (z | (z << 2)) & 0x09249249;

  return x | (y << 1) | (z << 2);
}

#endif
