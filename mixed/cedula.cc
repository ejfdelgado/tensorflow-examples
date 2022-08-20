#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "utils.h"
#include "utilsTensorCv.h"
#include "image2tensor.h"
#include "cedula.h"

// ./mixed /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/cedula/003.jpg ../tensor_python/models/cedulas_vaale-fp16.tflite -labels=../tensor_python/models/cedulas_vaale-fp16.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./
// ./mixed /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/cedula/002.jpg ../tensor_python/models/cedulas_vaale-fp16.tflite -labels=../tensor_python/models/cedulas_vaale-fp16.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./

using namespace cv;
using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char *argv[])
{
  cv::CommandLineParser parser(argc, argv,
                               "{@image       |            |The image file to segment or classify}"
                               "{@model       |            |The .tflite model file)}"
                               "{labels     l |            |The labels .txt file}"
                               "{imageType it |IMREAD_COLOR|Can be: IMREAD_COLOR IMREAD_GRAYSCALE}"
                               "{mode       m |FLOAT       |Can be: FLOAT CHAR}"
                               "{normalize  n |10          |Can be: 11 10 256}"
                               "{classIndex ci|-1          |Number of class index}"
                               "{scoreIndex si|-1          |Number of score index. Segmentation only.}"
                               "{yoloIndex yi |-1          |Number of yolo index. Segmentation only.}"
                               "{boxIndex   bi|-1          |Number of box index. Segmentation only.}"
                               "{threshold  th|0.6         |Threshold for score. Segmentation only.}"
                               "{sth          |0.5         |Threshold for class. Segmentation only.}"
                               "{nmsth        |0.45        |Threshold for nms. Segmentation only.}"
                               "{dpi          |70          |Dots per inch for text detection.}"
                               "{cedula       |            |Cedula.}"
                               "{cedtam       |850,550     |Cedula Tamanio W,H.}"
                               "{coords       |            |Coords.}"
                               "{ocrdir       |../mixed/   |Teseract folder dir where exists spa.traineddata.}"
                               "{outfolder    |            |The output folder, can be ./ Segmentation only.}");
  parser.printMessage();
  String imagePathString = parser.get<String>("@image");
  String modelPathString = parser.get<String>("@model");
  String labelsPathString = parser.get<String>("labels");
  String imageTypeString = parser.get<String>("imageType");
  String modeString = parser.get<String>("mode");
  int normalize = parser.get<int>("normalize");
  int classIndex = parser.get<int>("classIndex");
  int scoreIndex = parser.get<int>("scoreIndex");
  int boxIndex = parser.get<int>("boxIndex");
  int yoloIndex = parser.get<int>("yoloIndex");
  int dpi = parser.get<int>("dpi");
  float scoreThreshold = parser.get<float>("threshold");
  float sth = parser.get<float>("sth");
  float nmsth = parser.get<float>("nmsth");
  String outfolder = parser.get<String>("outfolder");
  String cedula = parser.get<String>("cedula");
  String cedtam = parser.get<String>("cedtam");
  String coords = parser.get<String>("coords");
  std::string TRAINED_FOLDER = parser.get<String>("ocrdir");

  cv::ImreadModes imageType = string2ImreadModesEnum(imageTypeString);

  // Load image
  cv::Mat image;
  image = cv::imread(imagePathString.c_str(), imageType);
  if (!image.data)
  {
    printf("No image data \n");
    return EXIT_FAILURE;
  }

  return 0;
}
