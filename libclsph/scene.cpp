#include "scene.h"

#include "util/tinyobj/tiny_obj_loader.h"

#include <iostream>
#include <cassert>
#include <cmath>

bool scene::load(std::string filename, float distFieldThreshold) {
  BB bbox;
  totalGridpoints = 0;
  face_count = 0;
  size_t sfaces;
  std::vector<tinyobj::shape_t> shapes;

  std::string err = tinyobj::LoadObj(
      shapes, (std::string("scenes/") + filename).c_str(), "scenes/");

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  std::cout << "Scene Loading - number of shapes in file [" << filename
            << "]: " << shapes.size() << std::endl;

  for (size_t i = 0; i < shapes.size(); ++i) {
    // TODO: handle multiple shapes properly
    indices = shapes[i].mesh.indices;
    if (indices.size() % 3 != 0) {
      std::cerr << "Meshes must be made of triangles only" << std::endl;
      return false;
    }

    sfaces = indices.size() / 3;
    face_count += sfaces;

    vertices = shapes[i].mesh.positions;

    // Pre-compute face normals
    for (size_t j = 0; j < sfaces; ++j) {
      size_t off = 3 * j;
      float nx, ny, nz;
      float ux, uy, uz, vx, vy, vz;

      bbox.maxx = std::numeric_limits<cl_int>::min();
      bbox.minx = std::numeric_limits<cl_int>::max();
      bbox.maxy = std::numeric_limits<cl_int>::min();
      bbox.miny = std::numeric_limits<cl_int>::max();
      bbox.maxz = std::numeric_limits<cl_int>::min();
      bbox.minz = std::numeric_limits<cl_int>::max();

      ux = vertices[3 * indices[off + 1] + 0] -
           vertices[3 * indices[off + 0] + 0];
      uy = vertices[3 * indices[off + 1] + 1] -
           vertices[3 * indices[off + 0] + 1];
      uz = vertices[3 * indices[off + 1] + 2] -
           vertices[3 * indices[off + 0] + 2];

      vx = vertices[3 * indices[off + 2] + 0] -
           vertices[3 * indices[off + 0] + 0];
      vy = vertices[3 * indices[off + 2] + 1] -
           vertices[3 * indices[off + 0] + 1];
      vz = vertices[3 * indices[off + 2] + 2] -
           vertices[3 * indices[off + 0] + 2];

      nx = uy * vz - uz * vy;
      ny = uz * vx - ux * vz;
      nz = ux * vy - uy * vx;

      float length = sqrt(nx * nx + ny * ny + nz * nz);
      float lengthu = sqrt(ux * ux + uy * uy + uz * uz);

      nx = nx / length;
      ny = ny / length;
      nz = nz / length;

      face_normals.push_back(nx);
      face_normals.push_back(ny);
      face_normals.push_back(nz);

      for (size_t k = 0; k < 3; k++) {
        if (vertices[3 * indices[off + k] + 0] > bbox.maxx)
          bbox.maxx = vertices[3 * indices[off + k] + 0];

        if (vertices[3 * indices[off + k] + 1] > bbox.maxy)
          bbox.maxy = vertices[3 * indices[off + k] + 1];

        if (vertices[3 * indices[off + k] + 2] > bbox.maxz)
          bbox.maxz = vertices[3 * indices[off + k] + 2];

        if (vertices[3 * indices[off + k] + 0] < bbox.minx)
          bbox.minx = vertices[3 * indices[off + k] + 0];

        if (vertices[3 * indices[off + k] + 1] < bbox.miny)
          bbox.miny = vertices[3 * indices[off + k] + 1];

        if (vertices[3 * indices[off + k] + 2] < bbox.minz)
          bbox.minz = vertices[3 * indices[off + k] + 2];
      }

      bbox.maxx += distFieldThreshold;
      bbox.maxy += distFieldThreshold;
      bbox.maxz += distFieldThreshold;
      bbox.minx -= distFieldThreshold;
      bbox.miny -= distFieldThreshold;
      bbox.minz -= distFieldThreshold;

      bbox.size_x =
          std::ceil((bbox.maxx - bbox.minx) / distFieldThreshold * 2) + 1;
      bbox.size_y =
          std::ceil((bbox.maxy - bbox.miny) / distFieldThreshold * 2) + 1;
      bbox.size_z =
          std::ceil((bbox.maxz - bbox.minz) / distFieldThreshold * 2) + 1;

      bbox.offset = totalGridpoints;
      totalGridpoints += bbox.size_x * bbox.size_y * bbox.size_z;
      ;

      bbs.push_back(bbox);

      float uux = ux / lengthu;
      float uuy = uy / lengthu;
      float uuz = uz / lengthu;

      float uvx = uuy * nz - uuz * ny;
      float uvy = uuz * nx - uux * nz;
      float uvz = uux * ny - uuy * nx;

      transforms.push_back(nx);
      transforms.push_back(ny);
      transforms.push_back(nz);
      transforms.push_back(-vertices[3 * indices[off + 0] + 0]);

      transforms.push_back(uvx);
      transforms.push_back(uvy);
      transforms.push_back(uvz);
      transforms.push_back(-vertices[3 * indices[off + 0] + 1]);

      transforms.push_back(uux);
      transforms.push_back(uuy);
      transforms.push_back(uuz);
      transforms.push_back(-vertices[3 * indices[off + 0] + 2]);

      rvertices.push_back(ux * uvx + uy * uvy + uz * uvz);
      rvertices.push_back(ux * uux + uy * uuy + uz * uuz);

      rvertices.push_back(uvx * vx + uvy * vy + uvz * vz);
      rvertices.push_back(uux * vx + uuy * vy + uuz * vz);
    }
  }
  return true;
}
