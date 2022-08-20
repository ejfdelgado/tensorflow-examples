cmake_minimum_required(VERSION 3.16)
project(mixed C CXX)

find_package( OpenCV REQUIRED )

find_package(Tesseract 4)
if (NOT Tesseract_FOUND)
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(Tesseract REQUIRED IMPORTED_TARGET tesseract lept)
    add_library(Tesseract::libtesseract ALIAS PkgConfig::Tesseract)
endif ()

SET(ABSL_PROPAGATE_CXX_STD ON)
SET(TENSOR_FLOW_SRC "/home/ejfdelgado/desarrollo/vaale/tensorflow")
#SET(TENSOR_FLOW_SRC "/home/ec2-user")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(
  "${TENSOR_FLOW_SRC}/tensorflow/tensorflow/lite"
  "${TENSOR_FLOW_SRC}/tensorflow/bazel-bin/tensorflow/lite"
  EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 11)
add_executable(mixed
  mixed.cc
  "${TENSOR_FLOW_SRC}/tensorflow/tensorflow/lite/profiling/root_profiler.cc"
)
add_executable(cedula
  cedula.cc
  "${TENSOR_FLOW_SRC}/tensorflow/tensorflow/lite/profiling/root_profiler.cc"
)
target_link_libraries(mixed
  tensorflow-lite
  ${OpenCV_LIBS}
  Tesseract::libtesseract
)
target_link_libraries(cedula
  tensorflow-lite
  ${OpenCV_LIBS}
  Tesseract::libtesseract
)
