cmake_minimum_required(VERSION 3.20)

# basic settings 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)
set(CUDA_ARCHITECTURES 86)
set(CUDA_SEPARABLE_COMPILATION ON)
set(ZS_ENABLE_OFB_ACCESS_CHECK ON)
set(ZS_ENABLE_JIT OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# add main targets 
project(tacipc LANGUAGES C CXX CUDA)
set_property(GLOBAL PROPERTY ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)
add_library(tacipc SHARED)
target_include_directories(tacipc PUBLIC include)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp src/*.hpp src/*.cu src/*.cuh)
target_sources(tacipc PUBLIC ${sources})

# add zpc
find_package(CUDAToolkit REQUIRED COMPONENTS cufft REQUIRED)
add_subdirectory(extern/zpc)
add_library(zshelper INTERFACE)
target_include_directories(zshelper INTERFACE external/zpc/include)
target_link_libraries(zshelper INTERFACE zensim)
target_compile_features(zshelper INTERFACE cuda_std_17)

# add igl 
add_subdirectory(extern/eigen)
set(Eigen3_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/eigen)
add_subdirectory(extern/libigl)
igl_include(glfw)
igl_include(imgui)

# add choldmod 
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(cholmod REQUIRED)

# add json
add_subdirectory(extern/json)

# handle link dependencies
target_link_libraries(tacipc PUBLIC zshelper igl::glfw igl::imgui Eigen3::Eigen cholmod nlohmann_json)
add_executable(main)
target_sources(main PUBLIC main.cu)
target_link_libraries(main PUBLIC tacipc)

# handle resources
# reference: https://stackoverflow.com/questions/34799916/copy-file-from-source-directory-to-binary-directory-using-cmake
# configure_file(${PROJECT_SOURCE_DIR}/resources/sponge.obj ${CMAKE_CURRENT_BINARY_DIR}/resources/sponge.obj COPYONLY)
# configure_file(${PROJECT_SOURCE_DIR}/resources/elastomer.msh ${CMAKE_CURRENT_BINARY_DIR}/resources/elastomer.msh COPYONLY)

add_definitions(-w)