#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// https://gist.github.com/WesleyCh3n/9653610bc384d15e0d9fc86b453751c4
// https://www.tensorflow.org/lite/examples
// cmake --build . -j 4
// Usage:
// ./mixed ../tensor_python/models/bee.jpg ../tensor_python/models/mobilenet/mobilenet_v2_1.0_224.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_quant_v1_224.txt -it=IMREAD_COLOR -m=FLOAT -n=10

using namespace cv;
using namespace std;

typedef cv::Point3_<char> PixelChar;
typedef cv::Point3_<float> PixelFloat;

std::vector<std::string> readLabelsFile(const char *path)
{
  std::vector<std::string> myvector;
  std::fstream newfile;
  newfile.open(path, ios::in);
  if (newfile.is_open())
  {
    string tp;
    while (getline(newfile, tp))
    {
      myvector.push_back(tp);
    }
    newfile.close();
  }

  return myvector;
}

template <typename T>
void genericNormalize(cv::Mat inputImg, uint nChanells, uint type)
{
  T *pixel = inputImg.ptr<T>(0, 0);
  int rows = inputImg.rows;
  int cols = inputImg.cols;
  for (uint i = 0; i < rows; i++)
  {
    for (uint j = 0; j < cols; j++)
    {
      for (uint z = 0; z < nChanells; z++)
      {
        const uint indice = (j * rows + i) * nChanells + z;
        if (type == 10)
        {
          pixel[indice] = (pixel[indice] / 255.0);
        }
        else if (type == 11)
        {
          pixel[indice] = ((pixel[indice] / 255.0) - 0.5) * 2.0;
        }
      }
    }
  }
}

template <typename T>
auto matPreprocess(cv::Mat src, uint width, uint height, uint nChanells, uint type) -> cv::Mat
{
  cv::Mat dst;
  if (type == 10 || type == 11)
  {
    src.convertTo(dst, CV_32FC3);
  }
  else if (type == 256)
  {
    dst = src.clone();
  }
  if (nChanells == 3)
  {
    // Only if RGB swap R with B
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
  }
  genericNormalize<T>(dst, nChanells, 10);
  cv::resize(dst, dst, cv::Size(width, height));
  return dst;
}

template <typename T>
void image2tensor(cv::Mat image, TfLiteTensor *input_tensor, uint WIDTH_M, uint HEIGHT_M, uint CHANNEL_M, uint type)
{
  cv::Mat inputImg;
  inputImg = matPreprocess<T>(image, WIDTH_M, HEIGHT_M, CHANNEL_M, type);
  float *pixel = inputImg.ptr<float>(0, 0);
  for (uint y = 0; y < HEIGHT_M; y++)
  {
    for (uint x = 0; x < WIDTH_M; x++)
    {
      for (uint z = 0; z < CHANNEL_M; z++)
      {
        uint xind = (y * WIDTH_M + x) * CHANNEL_M + z;
        uint xind2 = (x * HEIGHT_M + y) * CHANNEL_M + z;
        input_tensor->data.f[xind] = pixel[xind2];
      }
    }
  }
  return;
}

template <typename T>
auto cvtTensor(TfLiteTensor *tensor) -> vector<T>;

auto cvtTensor(TfLiteTensor *tensor) -> vector<float>
{
  int nelem = 1;
  for (int i = 0; i < tensor->dims->size; ++i)
    nelem *= tensor->dims->data[i];
  vector<float> data(tensor->data.f, tensor->data.f + nelem);
  return data;
}

cv::ImreadModes string2ImreadModesEnum(std::string str)
{
  static std::unordered_map<std::string, cv::ImreadModes> const table = {
      {"IMREAD_GRAYSCALE", cv::ImreadModes::IMREAD_GRAYSCALE},
      {"IMREAD_COLOR", cv::ImreadModes::IMREAD_COLOR}};
  auto it = table.find(str);
  if (it != table.end())
  {
    return it->second;
  }
  else
  {
    return cv::ImreadModes::IMREAD_COLOR;
  }
}

template <typename T>
void printTopClass(std::unique_ptr<tflite::Interpreter> *interpreter, uint classIndex, std::vector<string> class_names)
{
  TfLiteTensor *output_score = (*interpreter)->tensor((*interpreter)->outputs()[classIndex]);
  vector<T> score_vec = cvtTensor(output_score);
  T maxValue = 0;
  int maxIndex = -1;
  for (auto it = score_vec.begin(); it != score_vec.end(); it++)
  {
    int index = std::distance(score_vec.begin(), it);
    if (maxIndex == -1 || *it > maxValue)
    {
      maxValue = *it;
      maxIndex = index;
    }
  }
  uint classSize = class_names.size();
  std::cout << "[";
  std::cout << "{\"i\":" << maxIndex << ", ";
  if (maxIndex < classSize)
  {
    std::cout << "\"c\":\"" << class_names[maxIndex] << "\", ";
  }
  std::cout << "\"v\":" << maxValue << "},";
  std::cout << "]" << std::endl;
}

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char *argv[])
{
  cv::CommandLineParser parser(argc, argv,
                               "{@image      |            |The image file to segment or classify}"
                               "{@model      |            |The .tflite model file)}"
                               "{labels     l|            |The labels .txt file}"
                               "{imageType it|IMREAD_COLOR|Can be: IMREAD_COLOR IMREAD_GRAYSCALE}"
                               "{mode       m|FLOAT       |Can be: FLOAT CHAR}"
                               "{normalize  n|10          |Can be: 11 10 256}");
  parser.printMessage();
  String imagePathString = parser.get<String>("@image");
  String modelPathString = parser.get<String>("@model");
  String labelsPathString = parser.get<String>("labels");
  String imageTypeString = parser.get<String>("imageType");
  String modeString = parser.get<String>("mode");
  int normalize = parser.get<int>("normalize");

  cv::ImreadModes imageType = string2ImreadModesEnum(imageTypeString);

  // Load image
  cv::Mat image;
  image = cv::imread(imagePathString.c_str(), imageType);
  if (!image.data)
  {
    printf("No image data \n");
    return EXIT_FAILURE;
  }

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

  const uint HEIGHT_M = input_tensor->dims->data[1];
  const uint WIDTH_M = input_tensor->dims->data[2];
  const uint CHANNEL_M = input_tensor->dims->data[3];
  std::cout << "(" << HEIGHT_M << "x" << WIDTH_M << "x" << CHANNEL_M << ")" << std::endl;
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

  printTopClass<float>(&interpreter, 0, class_names);

  return 0;
}
