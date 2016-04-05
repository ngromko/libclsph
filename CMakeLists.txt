cmake_minimum_required(VERSION 2.6)
project(libclsph)

# Concat Files
function(cat IN_FILE OUT_FILE)
  file(READ ${IN_FILE} CONTENTS)
  file(APPEND ${OUT_FILE} "${CONTENTS}")
endfunction()

# User provided build options
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_OPENCL "use opencl" OFF)


set(GLFW_LOCATION "/home/nicholas/Documents/github/glfw-3.1.2")

#set(PROJECT_BINARY_DIR "bin")
#file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR})

# Set common compiler flags
set(COMPILE_FLAGS "--std=c++11 -g -Wfatal-errors -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-deprecated-declarations -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")

# Copy shader and properties files
file(COPY shaders DESTINATION ${PROJECT_BINARY_DIR}/examples)
file(COPY simulation_properties DESTINATION ${PROJECT_BINARY_DIR}/examples)
file(COPY scenes DESTINATION ${PROJECT_BINARY_DIR}/examples)
file(COPY fluid_properties DESTINATION ${PROJECT_BINARY_DIR}/examples)


# Check if OpenCL is present on the system
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(OpenCL)
if(OPENCL_FOUND)
    include_directories(${OPENCL_INCLUDE_DIRS})

    # Copy headers
    file(COPY libclsph/sph_simulation.h DESTINATION ${PROJECT_BINARY_DIR}/include/clsph)
    file(COPY libclsph/scene.h DESTINATION ${PROJECT_BINARY_DIR}/include/clsph)
    file(COPY libclsph/common DESTINATION ${PROJECT_BINARY_DIR}/include/clsph)
    file(COPY libclsph/file_save_delegates/houdini_file_saver.h DESTINATION ${PROJECT_BINARY_DIR}/include/clsph/file_save_delegates)
    file(COPY util DESTINATION ${PROJECT_BINARY_DIR}/include/clsph)

    # Copy kernels
    file(WRITE libclsph/kernels/sph.cl "" )
    cat( libclsph/kernels/common.cl libclsph/kernels/sph.cl)
    cat( libclsph/common/structures.h libclsph/kernels/sph.cl)
    cat( libclsph/common/util.h libclsph/kernels/sph.cl)
    cat( libclsph/kernels/smoothing.cl libclsph/kernels/sph.cl)
    cat( libclsph/kernels/grid.cl libclsph/kernels/sph.cl)
    cat( libclsph/kernels/forces.cl libclsph/kernels/sph.cl)
    cat( libclsph/kernels/collisionsv2.cl libclsph/kernels/sph.cl)
    cat( libclsph/kernels/sort.cl libclsph/kernels/sph.cl)
    cat( libclsph/kernels/advection.cl libclsph/kernels/sph.cl)
    cat( libclsph/kernels/sphb.cl libclsph/kernels/sph.cl)
    cat( libclsph/kernels/reduction.cl libclsph/kernels/sph.cl)

    file(COPY libclsph/kernels/sph.cl DESTINATION ${PROJECT_BINARY_DIR}/examples/kernels)

    include_directories(${PROJECT_BINARY_DIR}/include/clsph)

    # Build the main lib
    add_library(clsph
        libclsph/sph_simulation.cpp
        libclsph/file_save_delegates/houdini_file_saver.cpp
        util/cl_boilerplate.cpp
        util/houdini_geo/HoudiniFileDumpHelper.cpp
        libclsph/scene.cpp
        util/tinyobj/tiny_obj_loader.cc)

    set_target_properties(clsph PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
    set_target_properties(clsph PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

    target_link_libraries(clsph ${OPENCL_LIBRARIES})
endif(OPENCL_FOUND)

#Try to find CUDA
find_package(CUDA)

if(CUDA_FOUND)

    list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-g")
    SET(CUDA_PROPAGATE_HOST_FLAGS ON)

    # Copy headers
    file(COPY libcudasph/sph_simulation.cuh DESTINATION ${PROJECT_BINARY_DIR}/include/cudasph)
    file(COPY libcudasph/scene.cuh DESTINATION ${PROJECT_BINARY_DIR}/include/cudasph)
    file(COPY libcudasph/common DESTINATION ${PROJECT_BINARY_DIR}/include/cudasph)
    file(COPY libcudasph/file_save_delegates/houdini_file_saver.cuh DESTINATION ${PROJECT_BINARY_DIR}/include/cudasph/file_save_delegates)
    file(COPY util DESTINATION ${PROJECT_BINARY_DIR}/include/cudasph)

    # Copy kernels
    file(WRITE libcudasph/kernels/sph.cu "" )
    cat( libcudasph/kernels/operator.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/common.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/smoothing.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/grid.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/forces.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/collisionsv2.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/sort.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/advection.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/sphb.cu libcudasph/kernels/sph.cu)
    cat( libcudasph/kernels/reduction.cu libcudasph/kernels/sph.cu)

    #file(COPY libcudasph/kernels/sph.cu DESTINATION ${PROJECT_BINARY_DIR}/examples/kernels)

    include_directories(${PROJECT_BINARY_DIR}/include/cudasph)

    # Build the main lib
    cuda_add_library(cudasph
        libcudasph/sph_simulation.cu
        libcudasph/file_save_delegates/houdini_file_saver.cu
        util/houdini_geo/HoudiniFileDumpHelper.cpp
        libcudasph/scene.cu
        util/tinyobj/tiny_obj_loader.cc)

    set_target_properties(cudasph PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
    set_target_properties(cudasph PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif(CUDA_FOUND)

if(NOT CUDA_FOUND AND NOT OPENCL_FOUND)
message(FATAL_ERROR "OpenCL and CUDA not found")
endif(NOT CUDA_FOUND AND NOT OPENCL_FOUND)

if(BUILD_EXAMPLES)
    link_directories(${PROJECT_BINARY_DIR}/lib)

    if(OPENCL_FOUND)
        # Houdini output example
        add_executable(clsphparticles example/particles.cpp)
        set_target_properties(clsphparticles PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
        target_link_libraries(clsphparticles clsph)

        #find_package(OpenGL)
        #find_package(GLEW)
        #find_package(GLFW)

        # OpenGL rendered example requires OpenGL, GLEW and GLFW
        if(OPENGL_FOUND AND GLEW_FOUND AND GLFW_FOUND)
            include_directories(${OPENGL_INCLUDE_DIRS})
            include_directories(${GLEW_INCLUDE_DIRS})
            include_directories(${GLFW_INCLUDE_DIRS})

            add_executable(clsphopengl example/opengl.cpp)
            set_target_properties(clsphopengl PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)

            target_link_libraries(clsphopengl clsph)
            target_link_libraries(clsph ${OPENGL_LIBRARIES})
            target_link_libraries(clsph ${GLEW_LIBRARIES})
            target_link_libraries(clsph ${GLFW_LIBRARIES} -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -pthread )
        endif(OPENGL_FOUND AND GLEW_FOUND AND GLFW_FOUND)
    endif(OPENCL_FOUND)
    if(CUDA_FOUND)
        # Houdini output example
        cuda_add_executable(cudasphparticles example/particlescuda.cu)
        set_target_properties(cudasphparticles PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
        target_link_libraries(cudasphparticles cudasph)
    endif(CUDA_FOUND)
	
endif(BUILD_EXAMPLES)
