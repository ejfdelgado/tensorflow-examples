#include <cstdio>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// https://www.tensorflow.org/lite/examples
//  cmake --build . -j 4
//  Usage:
//  ./mixed /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/mobilenet/mobilenet_v1_1.0_224_quant.tflite /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/cat.jpg
//  ./mixed /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/yolov4-416-fp32.tflite /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/cat.jpg


using namespace cv;
using namespace std;

typedef cv::Point3_<float> Pixel;

const uint OUTDIM = 128;

void normalize(Pixel &pixel)
{
  pixel.x = ((pixel.x / 255.0) - 0.5) * 2.0;
  pixel.y = ((pixel.y / 255.0) - 0.5) * 2.0;
  pixel.z = ((pixel.z / 255.0) - 0.5) * 2.0;
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
  TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
  TfLiteTensor *output_score = interpreter->tensor(interpreter->outputs()[1]);

  const uint HEIGHT_M = input_tensor->dims->data[1];
  const uint WIDTH_M = input_tensor->dims->data[2];
  const uint CHANNEL_M = input_tensor->dims->data[3];

  cout << "HEIGHT_M=" << HEIGHT_M << endl;
  cout << "WIDTH_M=" << WIDTH_M << endl;
  cout << "CHANNEL_M=" << CHANNEL_M << endl;

  cout << "convert to float; BGR -> RGB" << endl;
  cv::Mat inputImg;
  image.convertTo(inputImg, CV_32FC3);
  cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2RGB);

  cout << "normalize to -1 & 1" << endl;
  Pixel *pixel = inputImg.ptr<Pixel>(0, 0);
  const Pixel *endPixel = pixel + inputImg.cols * inputImg.rows;
  for (; pixel != endPixel; pixel++)
    normalize(*pixel);

  cout << "resize image as model input" << endl;
  cv::resize(inputImg, inputImg, cv::Size(WIDTH_M, HEIGHT_M));

  float *inputLayer = interpreter->typed_input_tensor<float>(0);

  cout << "flatten rgb image to input layer" << endl;
  float *inputImg_ptr = inputImg.ptr<float>(0);
  cout << "memcpy..." << endl;
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

  vector<float> box_vec = cvtTensor(output_box);
  cout << "0" << endl;
  vector<float> score_vec = cvtTensor(output_score);

  cout << "1" << endl;

  vector<size_t> result_id;
  auto it = std::find_if(std::begin(score_vec), std::end(score_vec),
                         [](float i)
                         { return i > 0.6; });
  cout << "2" << endl;
  while (it != std::end(score_vec))
  {
    result_id.emplace_back(std::distance(std::begin(score_vec), it));
    it = std::find_if(std::next(it), std::end(score_vec),
                      [](float i)
                      { return i > 0.6; });
  }
  cout << "3" << endl;
  vector<cv::Rect> rects;
  vector<float> scores;
  for (size_t tmp : result_id)
  {
    const int cx = box_vec[4 * tmp];
    const int cy = box_vec[4 * tmp + 1];
    const int w = box_vec[4 * tmp + 2];
    const int h = box_vec[4 * tmp + 3];
    const int xmin = ((cx - (w / 2.f)) / WIDTH_M) * image.cols;
    const int ymin = ((cy - (h / 2.f)) / HEIGHT_M) * image.rows;
    const int xmax = ((cx + (w / 2.f)) / WIDTH_M) * image.cols;
    const int ymax = ((cy + (h / 2.f)) / HEIGHT_M) * image.rows;
    rects.emplace_back(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    scores.emplace_back(score_vec[tmp]);
  }
  cout << "4" << endl;
  vector<int> ids;
  cv::dnn::NMSBoxes(rects, scores, 0.6, 0.4, ids);
  for (int tmp : ids)
    cv::rectangle(image, rects[tmp], cv::Scalar(0, 255, 0), 3);
  cout << "5" << endl;
  cv::imwrite("./result.jpg", image);

  return 0;
}
