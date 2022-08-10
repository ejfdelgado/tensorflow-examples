#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// https://gilberttanner.com/blog/tflite-model-maker-object-detection/
// https://gist.github.com/WesleyCh3n/9653610bc384d15e0d9fc86b453751c4
// https://www.tensorflow.org/lite/examples
// cmake --build . -j 4
// Usage:
// ./mixed ../tensor_python/models/bee.jpg ../tensor_python/models/mobilenet/mobilenet_v2_1.0_224.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_quant_v1_224.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -si=0 -th=0.6
// ./mixed ../tensor_python/models/cat.jpg ../tensor_python/models/mobilenet/ssd_mobilenet_v1_1_metadata_1.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_v1.txt -it=IMREAD_COLOR -m=CHAR -n=256 -ci=1 -si=2 -bi=0 -th=0.6 -r=1
// ./mixed ../tensor_python/models/kite.jpg ../tensor_python/models/mobilenet/ssd_mobilenet_v1_1_metadata_1.tflite -labels=../tensor_python/models/mobilenet/labels_mobilenet_v1.txt -it=IMREAD_COLOR -m=CHAR -n=256 -ci=1 -si=2 -bi=0 -th=0.6 -r=1
// ./mixed ../tensor_python/models/fashion/shooe.png ../tensor_python/models/fashion/fashion.tflite -labels=../tensor_python/models/fashion/labels.txt -it=IMREAD_GRAYSCALE -m=FLOAT -n=10 -si=0

using namespace cv;
using namespace std;

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
    std::cout << "convertTo CV_32FC3" << std::endl;
    src.convertTo(dst, CV_32FC3);
  }
  else if (type == 256)
  {
    std::cout << "clone" << std::endl;
    dst = src.clone();
  }
  if (nChanells == 3)
  {
    // Only if RGB swap R with B
    std::cout << "cvtColor COLOR_BGR2RGB" << std::endl;
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
  }
  if (type == 10 || type == 11)
  {
    std::cout << "genericNormalize" << std::endl;
    genericNormalize<T>(dst, nChanells, type);
  }
  std::cout << "resize" << std::endl;
  cv::resize(dst, dst, cv::Size(width, height));
  return dst;
}

template <typename T>
void image2tensor(cv::Mat image, TfLiteTensor *input_tensor, uint WIDTH_M, uint HEIGHT_M, uint CHANNEL_M, uint type)
{
  cv::Mat inputImg;
  inputImg = matPreprocess<T>(image, WIDTH_M, HEIGHT_M, CHANNEL_M, type);
  memcpy(input_tensor->data.f, inputImg.ptr<T>(0),
         WIDTH_M * HEIGHT_M * CHANNEL_M * sizeof(T));
  /*
  T *pixel = inputImg.ptr<T>(0);
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
  */
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

struct DataIndexVal
{
  int index;
  float value;
};

bool compareByValue(const DataIndexVal &a, const DataIndexVal &b)
{
  return b.value < a.value;
}

template <typename T>
void printTopClass(std::unique_ptr<tflite::Interpreter> *interpreter, uint classIndex, std::vector<string> class_names, float scoreThreshold)
{
  TfLiteTensor *output_score = (*interpreter)->tensor((*interpreter)->outputs()[classIndex]);
  vector<T> score_vec = cvtTensor(output_score);
  T maxValue = 0;
  int maxIndex = -1;

  uint classSize = class_names.size();
  std::vector<DataIndexVal> passThreshold;
  for (auto it = score_vec.begin(); it != score_vec.end(); it++)
  {
    DataIndexVal nueva;
    nueva.index = std::distance(score_vec.begin(), it);
    ;
    nueva.value = *it;
    passThreshold.push_back(nueva);
  }
  std::sort(passThreshold.begin(), passThreshold.end(), compareByValue);
  std::vector<DataIndexVal> filteredVector;
  std::copy_if(passThreshold.begin(), passThreshold.end(), std::back_inserter(filteredVector), [scoreThreshold](DataIndexVal i)
               { return i.value >= scoreThreshold; });
  std::cout << "[";
  uint filteredVectorSize = filteredVector.size();
  for (auto it = filteredVector.begin(); it != filteredVector.end(); it++)
  {
    int index = std::distance(filteredVector.begin(), it);
    const uint maxIndex = (*it).index;
    const float maxValue = (*it).value;
    std::cout << "{\"i\":" << maxIndex << ", ";
    if (maxIndex < classSize)
    {
      std::cout << "\"c\":\"" << class_names[maxIndex] << "\", ";
    }
    std::cout << "\"v\":" << maxValue << "}";
    if (index < filteredVectorSize - 1)
    {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

template <typename T>
void printSegmented(std::unique_ptr<tflite::Interpreter> *interpreter, float scoreThreshold, uint boxIndex, uint scoreIndex, uint classIndex, std::vector<string> class_names, bool renderOutput, cv::Mat image)
{

  // Ubicaciones
  TfLiteTensor *output_box = (*interpreter)->tensor((*interpreter)->outputs()[boxIndex]);
  // Score
  TfLiteTensor *output_score = (*interpreter)->tensor((*interpreter)->outputs()[scoreIndex]);
  // Class
  TfLiteTensor *output_class = (*interpreter)->tensor((*interpreter)->outputs()[classIndex]);

  vector<float> box_vec = cvtTensor(output_box);
  vector<float> score_vec = cvtTensor(output_score);
  vector<float> class_vec = cvtTensor(output_class);

  vector<size_t> result_id;
  auto it = std::find_if(std::begin(score_vec), std::end(score_vec),
                         [scoreThreshold](float i)
                         { return i > scoreThreshold; });
  while (it != std::end(score_vec))
  {
    result_id.emplace_back(std::distance(std::begin(score_vec), it));
    it = std::find_if(std::next(it), std::end(score_vec),
                      [scoreThreshold](float i)
                      { return i > scoreThreshold; });
  }
  uint nFoundObjects = result_id.size();

  std::vector<cv::Rect> rects;
  std::vector<float> scores;

  uint classSize = class_names.size();
  std::cout << "[";
  for (size_t tmp : result_id)
  {
    // [arriba, izquierda, abajo, derecha]
    const float arriba = box_vec[4 * tmp];
    const float izquierda = box_vec[4 * tmp + 1];
    const float abajo = box_vec[4 * tmp + 2];
    const float derecha = box_vec[4 * tmp + 3];

    const int xmin = izquierda * image.cols;
    const int xmax = derecha * image.cols;
    const int ymin = arriba * image.rows;
    const int ymax = abajo * image.rows;

    std::cout << "{\"i\":" << class_vec[tmp] << ", ";
    std::cout << "\"xi\":\"" << xmin << "\", ";
    std::cout << "\"xf\":\"" << xmax << "\", ";
    std::cout << "\"yi\":\"" << ymin << "\", ";
    std::cout << "\"yf\":\"" << ymax << "\", ";
    if (class_vec[tmp] < classSize)
    {
      std::cout << "\"c\":\"" << class_names[class_vec[tmp]] << "\", ";
    }
    std::cout << "\"v\":" << score_vec[tmp] << "}";
    if (tmp < nFoundObjects - 1)
    {
      std::cout << ", ";
    }
    rects.emplace_back(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    scores.emplace_back(score_vec[tmp]);
  }
  std::cout << "]" << std::endl;

  if (renderOutput)
  {
    vector<int> ids;
    cv::dnn::NMSBoxes(rects, scores, 0.6, 0.4, ids);
    for (int tmp : ids)
      cv::rectangle(image, rects[tmp], cv::Scalar(0, 255, 0), 3);
    cv::imwrite("./result.jpg", image);
  }
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
                               "{render     r |0           |Render the output image. Segmentation only.}");
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
  bool renderOutput = parser.get<float>("render");

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

  if (scoreIndex >= 0)
  {
    printTopClass<float>(&interpreter, scoreIndex, class_names, scoreThreshold);
  }

  if (classIndex >= 0 && scoreIndex >= 0 && boxIndex >= 0)
  {
    printSegmented<float>(&interpreter, scoreThreshold, boxIndex, scoreIndex, classIndex, class_names, renderOutput, image);
  }

  return 0;
}
