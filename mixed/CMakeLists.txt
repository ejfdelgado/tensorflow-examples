cmake_minimum_required(VERSION 3.16)
project(mixed C CXX)

# cmake ../mixed
# cmake --build . -j 4
# vi ../mixed/CMakeLists.txt
# node /home/ec2-user/tensorflow-examples/utils/shared-libs.js cedula cedula.zip
# Run first AWS credentials /home/ejfdelgado/desarrollo/amazon/rootkey.csv
# aws s3api put-object --bucket ejfdelgado-simple --key libs/cedula.zip --body cedula.zip

# /home/ec2-user/tensorflow-examples/mixed-build
# ldconfig -p | grep lept
# ldconfig -p | grep libtesseract

find_package( OpenCV REQUIRED )
find_package(Tesseract 4 REQUIRED)

SET(modo "AWS")

IF(modo STREQUAL "LOCAL")
  message("Running Local...")
  # export OpenCV_DIR=/home/ejfdelgado/desarrollo/vaale/build
  # export Leptonica_DIR=/home/ejfdelgado/desarrollo/tesseract/leptonica-1.79.0/build
  # export Tesseract_DIR=/home/ejfdelgado/desarrollo/tesseract/tesseract-4.1.2/build
  SET(TENSOR_FLOW_SRC "/home/ejfdelgado/desarrollo/vaale/tensorflow")
  find_package(Leptonica REQUIRED)
ELSE()
  message("Running AMI...")
  # export Leptonica_DIR=/home/ec2-user/tesseract/leptonica-1.79.0/build
  # export Tesseract_DIR=/home/ec2-user/tesseract/tesseract-4.1.2/build
  # export OpenCV_DIR=/home/ec2-user/opencv-build
  # SET(CMAKE_PREFIX_PATH "/usr/local/lib/pkgconfig")
  find_package(PkgConfig REQUIRED)
  SET(TENSOR_FLOW_SRC "/home/ec2-user")
  pkg_check_modules(Leptonica REQUIRED lept REQUIRED)
  link_directories( ${Leptonica_LIBRARY_DIRS} )
ENDIF()


#if (NOT Tesseract_FOUND)
#  find_package(PkgConfig REQUIRED)
#  pkg_check_modules(Tesseract REQUIRED IMPORTED_TARGET tesseract lept)
#  add_library(Tesseract::libtesseract ALIAS PkgConfig::Tesseract)
#endif ()

SET(ABSL_PROPAGATE_CXX_STD ON)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Tesseract_INCLUDE_DIRS} )
include_directories( ${Leptonica_INCLUDE_DIRS} )

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
add_executable(solvePnP
  solvePnP.cc
)

target_link_libraries(mixed
  tensorflow-lite
  ${OpenCV_LIBS}
  # Tesseract::libtesseract
  ${Leptonica_LIBRARIES}
  ${Tesseract_LIBRARIES}
)
# https://forums.raspberrypi.com/viewtopic.php?t=292506
target_link_libraries(cedula
  tensorflow-lite
  ${OpenCV_LIBS}
  # Tesseract::libtesseract
  ${Leptonica_LIBRARIES}
  ${Tesseract_LIBRARIES}
)
target_link_libraries(solvePnP
  ${OpenCV_LIBS}
)
