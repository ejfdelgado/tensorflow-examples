# mkdir minimal-tf-build
# cd minimal-tf-build
# cmake ../minimal-tf
# cmake --build . -j 4

cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

SET(ABSL_PROPAGATE_CXX_STD ON)
SET(TENSOR_FLOW_SRC "/home/ejfdelgado/desarrollo/vaale/tensorflow")

add_subdirectory(
  "${TENSOR_FLOW_SRC}/tensorflow/tensorflow/lite"
  "${TENSOR_FLOW_SRC}/tensorflow/bazel-bin/tensorflow/lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 11)
add_executable(minimal
  minimal.cc
  "${TENSOR_FLOW_SRC}/tensorflow/tensorflow/lite/profiling/root_profiler.cc"
)
target_link_libraries(minimal
  tensorflow-lite
)
