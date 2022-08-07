#include <cstdio>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// https://www.tensorflow.org/lite/examples/object_detection/overview#get_started
// https://gist.github.com/WesleyCh3n/9653610bc384d15e0d9fc86b453751c4
// https://www.tensorflow.org/lite/examples
// cmake --build . -j 6
// Usage:
// ./segmentation2 /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/mobilenet/ssd_mobilenet_v1_1_metadata_1.tflite /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/cat.jpg
// ./segmentation2 /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/mobilenet/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite /home/ejfdelgado/desarrollo/tensorflow-examples/tensor_python/models/cat.jpg

using namespace cv;
using namespace std;

typedef cv::Point3_<char> PixelChar;
typedef cv::Point3_<float> PixelFloat;

void normalize_1_1(PixelFloat &pixel)
{
  pixel.x = ((pixel.x / 255.0) - 0.5) * 2.0;
  pixel.y = ((pixel.y / 255.0) - 0.5) * 2.0;
  pixel.z = ((pixel.z / 255.0) - 0.5) * 2.0;
}

void normalize_0_1(PixelFloat &pixel)
{
  pixel.x = (pixel.x / 255.0);
  pixel.y = (pixel.y / 255.0);
  pixel.z = (pixel.z / 255.0);
}

auto matPreprocessFloat(cv::Mat src, uint width, uint height) -> cv::Mat
{
  // convert to float; BGR -> RGB
  cv::Mat dst;
  src.convertTo(dst, CV_32FC3);
  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

  // normalize
  PixelFloat *pixel = dst.ptr<PixelFloat>(0, 0);
  const PixelFloat *endPixel = pixel + dst.cols * dst.rows;
  for (; pixel != endPixel; pixel++)
    normalize_0_1(*pixel);
  // resize image as model input
  cv::resize(dst, dst, cv::Size(width, height));
  return dst;
}

auto matPreprocessChar(cv::Mat src, uint width, uint height) -> cv::Mat
{
  cv::Mat dst;
  dst = src.clone();
  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
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
  // printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);

  const uint HEIGHT_M = input_tensor->dims->data[1];
  const uint WIDTH_M = input_tensor->dims->data[2];
  const uint CHANNEL_M = input_tensor->dims->data[3];

  cout << "HEIGHT_M=" << HEIGHT_M << endl;
  cout << "WIDTH_M=" << WIDTH_M << endl;
  cout << "CHANNEL_M=" << CHANNEL_M << endl;

  cv::Mat inputImg = matPreprocessChar(image, WIDTH_M, HEIGHT_M);
  memcpy(input_tensor->data.f, inputImg.ptr<char>(0),
         WIDTH_M * HEIGHT_M * CHANNEL_M * sizeof(char));

  cout << "Run inference" << endl;
  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  // printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  cout << "Get Output Tensor..." << endl;

  // Ubicaciones
  TfLiteTensor *output_box = interpreter->tensor(interpreter->outputs()[0]);
  // Score
  TfLiteTensor *output_score = interpreter->tensor(interpreter->outputs()[2]);
  // Class
  TfLiteTensor *output_class = interpreter->tensor(interpreter->outputs()[1]);

  vector<float> box_vec = cvtTensor(output_box);
  vector<float> score_vec = cvtTensor(output_score);
  vector<float> class_vec = cvtTensor(output_class);

  vector<size_t> result_id;
  auto it = std::find_if(std::begin(score_vec), std::end(score_vec),
                         [](float i)
                         { return i > 0.6; });
  while (it != std::end(score_vec))
  {
    result_id.emplace_back(std::distance(std::begin(score_vec), it));
    it = std::find_if(std::next(it), std::end(score_vec),
                      [](float i)
                      { return i > 0.6; });
  }

  cout << "Objects detected: " << result_id.size() << endl;

  vector<cv::Rect> rects;
  vector<float> scores;
  for (size_t tmp : result_id)
  {
    // [arriba, izquierda, abajo, derecha]
    const float arriba = box_vec[4 * tmp];
    const float izquierda = box_vec[4 * tmp + 1];
    const float abajo = box_vec[4 * tmp + 2];
    const float derecha = box_vec[4 * tmp + 3];
    cout << "clase:" << class_vec[tmp] << ", coords: " << arriba << "," << izquierda << "," << abajo << "," << derecha << endl;

    const int xmin = izquierda * image.cols;
    const int xmax = derecha * image.cols;
    const int ymin = arriba * image.rows;
    const int ymax = abajo * image.rows;

    rects.emplace_back(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    scores.emplace_back(score_vec[tmp]);
  }

  vector<int> ids;
  cv::dnn::NMSBoxes(rects, scores, 0.6, 0.4, ids);
  for (int tmp : ids)
    cv::rectangle(image, rects[tmp], cv::Scalar(0, 255, 0), 3);
  cv::imwrite("./result.jpg", image);

  return 0;
}
