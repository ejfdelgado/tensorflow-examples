#!/bin/bash
set -e

sudo amazon-linux-extras install -y java-openjdk11
sudo yum update
sudo yum install -y gcc gcc-c++ patch python3 git
sudo yum install -y g++ wget unzip numpy

sudo yum remove cmake

wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh
chmod +x cmake-3.24.0-linux-x86_64.sh
./cmake-3.24.0-linux-x86_64.sh --skip-license
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/cmake /usr/bin/cmake
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/ccmake /usr/bin/ccmake
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/cpack /usr/bin/cpack
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/ctest /usr/bin/ctest

wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh
chmod +x bazel-5.1.1-installer-linux-x86_64.sh
./bazel-5.1.1-installer-linux-x86_64.sh --user
#source /home/ec2-user/.bazel/bin/bazel-complete.bash

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout 2.9.0

pip3 install numpy

bazel build --jobs 4 -c opt --config=monolithic //tensorflow:libtensorflow_cc.so

bazel build --jobs 4 -c opt //tensorflow:libtensorflow_framework.so

bazel build --jobs 4 //tensorflow/lite:libtensorflowlite.so

cd ..

git clone https://github.com/ejfdelgado/tensorflow-examples.git

cd tensorflow-examples

mkdir minimal-tf-build

cd minimal-tf-build

cmake ../minimal-tf

cmake --build . -j 4