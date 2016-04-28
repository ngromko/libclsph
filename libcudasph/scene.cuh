#ifndef _SCENE_H_
#define _SCENE_H_
#include "common/structures.cuh"

#include <vector>
#include <string>

class scene {
 public:
  bool load(std::string filename, float distFieldThreshold);

  unsigned int face_count;
  std::vector<float> face_normals;
  std::vector<float> vertices;
  std::vector<unsigned int> indices;
  std::vector<BB> bbs;
  std::vector<float> transforms;
  std::vector<float> rvertices;
  unsigned int totalGridpoints;
};

#endif
