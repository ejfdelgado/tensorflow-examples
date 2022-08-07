#!/bin/bash
set -e

# Run first AWS credentials /home/ejfdelgado/desarrollo/amazon/rootkey.csv

# Install dependencies
sudo amazon-linux-extras install -y java-openjdk11
sudo yum update
sudo yum install -y gcc gcc-c++ patch python3 git
sudo yum install -y g++ wget unzip numpy
pip3 install numpy

# Install cmake 3.24.0
wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh
chmod +x cmake-3.24.0-linux-x86_64.sh
./cmake-3.24.0-linux-x86_64.sh --skip-license
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/cmake /usr/bin/cmake
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/ccmake /usr/bin/ccmake
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/cpack /usr/bin/cpack
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/ctest /usr/bin/ctest

# Install bazel 5.1.1
wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh
chmod +x bazel-5.1.1-installer-linux-x86_64.sh
./bazel-5.1.1-installer-linux-x86_64.sh --user
#source /home/ec2-user/.bazel/bin/bazel-complete.bash

# Install node
wget https://rpm.nodesource.com/setup_16.x
chmod +x setup_16.x
sudo ./setup_16.x
sudo yum install -y nodejs

# Install opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/heads/4.x.zip
mkdir opencv
unzip opencv.zip -d opencv
mkdir opencv-build
cd opencv-build
cmake  ../opencv/opencv-4.x
make -j 4
export OpenCV_DIR=/home/ec2-user/opencv-build

cd ..

# Build Tensorflow with Bazel
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout 2.9.0
bazel build --jobs 4 -c opt --config=monolithic //tensorflow:libtensorflow_cc.so
bazel build --jobs 4 -c opt //tensorflow:libtensorflow_framework.so
bazel build --jobs 4 //tensorflow/lite:libtensorflowlite.so

cd ..

# Clone own project
git clone https://github.com/ejfdelgado/tensorflow-examples.git
cd tensorflow-examples

# Build tensorflow minimal
mkdir minimal-tf-build
cd minimal-tf-build
cmake ../minimal-tf
cmake --build . -j 4
./minimal ../tensor_python/models/petals.tflite 5.0 3.2 1.2 0.2
node /home/ec2-user/tensorflow-examples/utils/shared-libs.js /home/ec2-user/tensorflow-examples/minimal-tf-build/minimal minimal.zip
aws s3api put-object --bucket ejfdelgado-simple --key libs/minimal.zip --body minimal.zip

cd ..

# Build opencv minimal
mkdir segmentation-build
cd segmentation-build
cmake ../segmentation
cmake --build . -j 4
