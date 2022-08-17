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

// https://gilberttanner.com/blog/tflite-model-maker-object-detection/
// https://gist.github.com/WesleyCh3n/9653610bc384d15e0d9fc86b453751c4
// https://www.tensorflow.org/lite/examples
// cmake ../mixed
// cmake --build . -j 4
// node ../utils/shared-libs.js ./mixed
// Usage:
// ./mixed ../tensor_python/models/bee.jpg ../tensor_python/models/mobilenet/mobilenet_v2_1.0_224.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_quant_v1_224.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -si=0 -th=0.6
// ./mixed ../tensor_python/models/cat.jpg ../tensor_python/models/mobilenet/ssd_mobilenet_v1_1_metadata_1.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_v1.txt -it=IMREAD_COLOR -m=CHAR -n=256 -ci=1 -si=2 -bi=0 -th=0.6 -outfolder=./
// ./mixed ../tensor_python/models/kite.jpg ../tensor_python/models/mobilenet/ssd_mobilenet_v1_1_metadata_1.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_v1.txt -it=IMREAD_COLOR -m=CHAR -n=256 -ci=1 -si=2 -bi=0 -th=0.6 -outfolder=./
// ./mixed ../tensor_python/models/fashion/shooe.png ../tensor_python/models/fashion/fashion.tflite -labels=../tensor_python/models/fashion/labels.txt -it=IMREAD_GRAYSCALE -m=FLOAT -n=10 -si=0
// ./mixed ../tensor_python/models/cat.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.25
// ./mixed ../tensor_python/models/bee.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.25
// ./mixed ../tensor_python/models/zebra.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.25

using namespace cv;
using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net, uint INPUT_WIDTH, uint INPUT_HEIGHT)
{
  // Convert to blob.
  cv::Mat blob;
  cv::dnn::blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

  net.setInput(blob);

  // Forward propagate.
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  return outputs;
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
                               "{boxIndex   bi|-1          |Number of box index. Segmentation only.}"
                               "{threshold  th|0.6         |Threshold for score. Segmentation only.}"
                               "{outfolder    |            |The output folder. Segmentation only.}");
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
  float scoreThreshold = parser.get<float>("threshold");
  String outfolder = parser.get<String>("outfolder");

  cv::ImreadModes imageType = string2ImreadModesEnum(imageTypeString);

  // Load image
  cv::Mat image;
  image = cv::imread(imagePathString.c_str(), imageType);
  if (!image.data)
  {
    printf("No image data \n");
    return EXIT_FAILURE;
  }
  std::cout << "{";
  std::cout << "\"width\":" << image.cols << ", ";
  std::cout << "\"height\":" << image.rows << ", ";
  std::cout << "\"chanells\":" << image.channels();
  std::cout << "}" << std::endl;

  // Load labels path
  std::vector<string> class_names;
  if (labelsPathString.compare("") != 0)
  {
    std::cout << "Reading labels..." << std::endl;
    class_names = readLabelsFile(labelsPathString.c_str());
  }

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(modelPathString.c_str());
  TFLITE_MINIMAL_CHECK(model != nullptr);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);
  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());
  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);

  const uint WHAT_IS_IT = input_tensor->dims->data[0]; // it is 1
  const uint HEIGHT_M = input_tensor->dims->data[1];
  const uint WIDTH_M = input_tensor->dims->data[2];
  const uint CHANNEL_M = input_tensor->dims->data[3];
  std::cout << "(" << WHAT_IS_IT << "x" << HEIGHT_M << "x" << WIDTH_M << "x" << CHANNEL_M << ")" << std::endl;
  cv::Mat inputImg;
  if (modeString.compare("CHAR") == 0)
  {
    image2tensor<char>(image, input_tensor, WIDTH_M, HEIGHT_M, CHANNEL_M, normalize);
  }
  else if (modeString.compare("FLOAT") == 0)
  {
    image2tensor<float>(image, input_tensor, WIDTH_M, HEIGHT_M, CHANNEL_M, normalize);
  }

  std::cout << "Run inference..." << std::endl;
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  if (scoreIndex >= 0)
  {
    printTopClass<float>(&interpreter, scoreIndex, class_names, scoreThreshold);
  }

  if (classIndex >= 0 && scoreIndex >= 0 && boxIndex >= 0)
  {
    printSegmented<float>(&interpreter, scoreThreshold, boxIndex, scoreIndex, classIndex, class_names, outfolder, image);
  }

/*
  std::vector<cv::Rect> res = printYoloV5(
      &interpreter,
      image,
      class_names,
      scoreThreshold, // CONFIDENCE_THRESHOLD=0.45
      0.5,            // SCORE_THRESHOLD=0.5 para la clase
      WIDTH_M,
      HEIGHT_M);
      */

  return 0;
}
