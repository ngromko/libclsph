#include "houdini_file_saver.cuh"
#include <sstream>
#include <iostream>
#include <fstream>
#include <iosfwd>
#include <iomanip>
#include <string>

#include "util/houdini_geo/HoudiniFileDumpHelper.h"

#ifdef USE_PARTIO
#include "util/partio/PartioFunctions.h"
#endif

#define OUTPUT_FILE_NAME "frames/frame"

// Taken from http://www.cplusplus.com/forum/general/15952/
std::string ZeroPadNumber(int num) {
  std::ostringstream ss;
  ss << std::setw(9) << std::setfill('0') << num;
  std::string result = ss.str();
  if (result.length() > 7) {
    result.erase(0, result.length() - 7);
  }
  return result;
}

int houdini_file_saver::writeFrameToFile(
    particle* particles, const simulation_parameters& parameters) {
#ifndef USE_PARTIO

  // TODO Find a way to use the same struct for the simulation and writing to
  // file
  houdini_Particle* houdini_Particles =
      new houdini_Particle[parameters.particles_count];

  std::stringstream ss;
  ss << frames_folder_prefix << OUTPUT_FILE_NAME << ZeroPadNumber(++frame_count)
     << ".geo";
  std::string fileName = ss.str();

  for (unsigned int i = 0; i < parameters.particles_count; i++) {
    houdini_Particles[i].px = particles[i].position.x;
    houdini_Particles[i].py = particles[i].position.y;
    houdini_Particles[i].pz = particles[i].position.z;
    houdini_Particles[i].vx = particles[i].velocity.x;
    houdini_Particles[i].vy = particles[i].velocity.y;
    houdini_Particles[i].vz = particles[i].velocity.z;
    // TODO Use actual colors
    houdini_Particles[i].colorR =
        particles[i].density > 1000.f && particles[i].density <= 2000.f
            ? (particles[i].density - 1000.f) / 1000.f
            : 0.f;
    houdini_Particles[i].colorG =
        particles[i].density >= 0.f && particles[i].density < 1000.f
            ? 1.f - (particles[i].density) / 1000.f
            : 0.f;
    houdini_Particles[i].colorB =
        particles[i].density >= 500.f && particles[i].density <= 1000.f
            ? (particles[i].density - 500.f) / 500.f
            : particles[i].density >= 1000.f && particles[i].density <= 1500.f
                  ? 1.f - (particles[i].density - 1000.f) / 500.f
                  : 0.f;
    houdini_Particles[i].mass = parameters.particle_mass;
  }

  std::ofstream fichier(fileName, std::ios::out | std::ios::trunc);
  if (fichier.is_open()) {
    ConcreteDataProvider dataProvider(houdini_Particles,
                                      parameters.particles_count);
    HoudiniFileDumpParticles partsDumper(&dataProvider);
    partsDumper.dump(fichier);

    fichier.close();
  } else {
    std::cerr << "Error while writing to " << fileName << std::endl;
  }

  delete[] houdini_Particles;

#else

  std::stringstream ss;
  ss << frames_folder_prefix << OUTPUT_FILE_NAME << ZeroPadNumber(++frame_count)
     << ".bgeo";
  std::string fileName = ss.str();

  Partio::ParticlesDataMutable* data = makeData(particles, parameters);
  Partio::write(fileName.c_str(), *data);
  data->release();

#endif

  return 0;
}
