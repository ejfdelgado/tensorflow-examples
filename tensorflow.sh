#!/bin/bash
set -e

sudo amazon-linux-extras install -y java-openjdk11
sudo yum update
sudo yum install -y gcc gcc-c++ patch python3 git
sudo yum install -y cmake g++ wget unzip numpy

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
