#include <cstdio>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// https://gist.github.com/WesleyCh3n/9653610bc384d15e0d9fc86b453751c4
// https://www.tensorflow.org/lite/examples
//  cmake --build . -j 4
//  Usage:
//  ./mixed /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/mobilenet/mobilenet_v1_1.0_224_quant.tflite /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/bee.jpg
//  ./mixed /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/yolov4-416-fp32.tflite /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/cat.jpg

using namespace cv;
using namespace std;

typedef cv::Point3_<float> Pixel;


void normalize(Pixel &pixel)
{
  pixel.x = ((pixel.x / 255.0) - 0.5) * 2.0;
  pixel.y = ((pixel.y / 255.0) - 0.5) * 2.0;
  pixel.z = ((pixel.z / 255.0) - 0.5) * 2.0;
}

/*
void normalize(Pixel &pixel)
{
  pixel.x = (pixel.x / 255.0);
  pixel.y = (pixel.y / 255.0);
  pixel.z = (pixel.z / 255.0);
}
*/

auto matPreprocess(cv::Mat src, uint width, uint height) -> cv::Mat
{
  // convert to float; BGR -> RGB
  cv::Mat dst;
  src.convertTo(dst, CV_32FC3);
  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

  // normalize to -1 & 1
  Pixel *pixel = dst.ptr<Pixel>(0, 0);
  const Pixel *endPixel = pixel + dst.cols * dst.rows;
  for (; pixel != endPixel; pixel++)
    normalize(*pixel);

  // resize image as model input
  cv::resize(dst, dst, cv::Size(width, height));
  return dst;
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

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    fprintf(stderr, "minimal <tflite model> <Image>\n");
    return -1;
  }

  cv::Mat image;
  image = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (!image.data)
  {
    printf("No image data \n");
    return -1;
  }

  const char *filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  TfLiteTensor *output_score = interpreter->tensor(interpreter->outputs()[0]);

  const uint HEIGHT_M = input_tensor->dims->data[1];
  const uint WIDTH_M = input_tensor->dims->data[2];
  const uint CHANNEL_M = input_tensor->dims->data[3];

  cout << "HEIGHT_M=" << HEIGHT_M << endl;
  cout << "WIDTH_M=" << WIDTH_M << endl;
  cout << "CHANNEL_M=" << CHANNEL_M << endl;

  cv::Mat inputImg = matPreprocess(image, WIDTH_M, HEIGHT_M);
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0),
         WIDTH_M * HEIGHT_M * CHANNEL_M * sizeof(float));

  cout << "Run inference" << endl;
  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  cout << "Get Output Tensor..." << endl;
  float *outputLayer = interpreter->typed_output_tensor<float>(0);

  cout << "Print final result..." << endl;

  vector<float> score_vec = cvtTensor(output_score);
  float maxValue = 0;
  int maxIndex = -1;
  for (auto it = score_vec.begin(); it != score_vec.end(); it++)
  {
    int index = std::distance(score_vec.begin(), it);
    if (*it > maxValue)
    {
      maxValue = *it;
      maxIndex = index;
    }
    // std::cout << "Element " << *it << " found at index " << index << std::endl;
  }
  std::cout << "maxElementIndex:" << maxIndex << ", maxElement:" << maxValue << '\n';

  /*
  int maxElementIndex = std::max_element(score_vec.begin(), score_vec.end()) - score_vec.begin();
  int maxElement = *std::max_element(score_vec.begin(), score_vec.end());

  int minElementIndex = std::min_element(score_vec.begin(), score_vec.end()) - score_vec.begin();
  int minElement = *std::min_element(score_vec.begin(), score_vec.end());

  std::cout << "maxElementIndex:" << maxElementIndex << ", maxElement:" << maxElement << '\n';
  std::cout << "minElementIndex:" << minElementIndex << ", minElement:" << minElement << '\n';
  */

  return 0;
}
