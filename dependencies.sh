#!/bin/bash
set -e

# Run first AWS credentials /home/ejfdelgado/desarrollo/amazon/rootkey.csv

# Install dependencies
sudo amazon-linux-extras install -y java-openjdk11
sudo yum update
sudo yum install -y gcc gcc-c++ patch python3 git
sudo yum install -y g++ wget unzip numpy
pip3 install numpy

# Install node
wget https://rpm.nodesource.com/setup_16.x
chmod +x setup_16.x
sudo ./setup_16.x
sudo yum install -y nodejs

# Install cmake 3.24.0
wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh
chmod +x cmake-3.24.0-linux-x86_64.sh
./cmake-3.24.0-linux-x86_64.sh --skip-license
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/cmake /usr/bin/cmake
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/ccmake /usr/bin/ccmake
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/cpack /usr/bin/cpack
sudo ln -s /home/ec2-user/cmake-3.24.0-linux-x86_64/bin/ctest /usr/bin/ctest
# Esto es nuevo con el ocr
# tesseract-ocr 4.1.1-2.1build1
# libtesseract-dev 4.1.1-2.1build1
# sudo yum install tesseract-ocr libtesseract-dev

# Build Leptonica -> 1.79.0
wget http://www.leptonica.org/source/leptonica-1.79.0.tar.gz .
tar -zxvf leptonica-1.79.0.tar.gz
cd leptonica-1.79.0
#./configure --prefix=/usr/local/leptonica-1.79.0
mkdir build && cd build
cmake ..
make
sudo make install
export PKG_CONFIG_PATH=/usr/local/leptonica-1.79.0/lib/pkgconfig
cd .. && cd ..

# Build Leptonica -> 1.74.2
#mkdir tesseract
#cd tesseract
#wget https://github.com/DanBloomberg/leptonica/archive/refs/tags/v1.74.3.zip
#unzip v1.74.3.zip
#cd leptonica-1.74.3
#mkdir build && cd build
#cmake ..
#make
#sudo make install
#cd .. && cd ..

# Build tesseract 4.1.2
wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/4.1.2.zip
unzip 4.1.2.zip
cd tesseract-4.1.2
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig
cd .. && cd ..
# /usr/local/lib/pkgconfig/tesseract.pc
# export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
# cd /usr/local/lib/pkgconfig
# cd /home/ec2-user/tensorflow-examples/mixed-build

# Install bazel 5.1.1
wget https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-installer-linux-x86_64.sh
chmod +x bazel-5.1.1-installer-linux-x86_64.sh
./bazel-5.1.1-installer-linux-x86_64.sh --user
#source /home/ec2-user/.bazel/bin/bazel-complete.bash

# Install opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/heads/4.x.zip
mkdir opencv
unzip opencv.zip -d opencv
mkdir opencv-build
cd opencv-build
cmake  ../opencv/opencv-4.x
make -j 4
#export OpenCV_DIR=/home/ejfdelgado/desarrollo/vaale/build
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
node ../utils/shared-libs.js minimal minimal.zip
aws s3api put-object --bucket ejfdelgado-simple --key libs/minimal.zip --body minimal.zip

cd ..

# Build opencv mixed
mkdir mixed-build
cd mixed-build
cmake ../mixed
cmake --build . -j 4
./mixed ../tensor_python/models/bee.jpg ../tensor_python/models/mobilenet/mobilenet_v2_1.0_224.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_quant_v1_224.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -si=0 -th=0.6
node ../utils/shared-libs.js mixed mixed.zip
aws s3api put-object --bucket ejfdelgado-simple --key libs/mixed.zip --body mixed.zip

