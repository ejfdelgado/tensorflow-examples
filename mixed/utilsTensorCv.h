#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

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
auto cvtTensor(TfLiteTensor *tensor) -> std::vector<T>;

auto cvtTensor(TfLiteTensor *tensor) -> std::vector<float>
{
  int nelem = 1;
  for (int i = 0; i < tensor->dims->size; ++i)
    nelem *= tensor->dims->data[i];
  std::vector<float> data(tensor->data.f, tensor->data.f + nelem);
  return data;
}

template <typename T>
void printTopClass(std::unique_ptr<tflite::Interpreter> *interpreter, uint classIndex, std::vector<std::string> class_names, float scoreThreshold)
{
  TfLiteTensor *output_score = (*interpreter)->tensor((*interpreter)->outputs()[classIndex]);
  std::vector<T> score_vec = cvtTensor(output_score);
  T maxValue = 0;
  int maxIndex = -1;

  uint classSize = class_names.size();
  std::vector<DataIndexVal> passThreshold;
  for (auto it = score_vec.begin(); it != score_vec.end(); it++)
  {
    DataIndexVal nueva;
    nueva.index = std::distance(score_vec.begin(), it);
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
void printSegmented(std::unique_ptr<tflite::Interpreter> *interpreter, float scoreThreshold, uint boxIndex, uint scoreIndex, uint classIndex, std::vector<std::string> class_names, bool renderOutput, cv::Mat image)
{

  // Ubicaciones
  TfLiteTensor *output_box = (*interpreter)->tensor((*interpreter)->outputs()[boxIndex]);
  // Score
  TfLiteTensor *output_score = (*interpreter)->tensor((*interpreter)->outputs()[scoreIndex]);
  // Class
  TfLiteTensor *output_class = (*interpreter)->tensor((*interpreter)->outputs()[classIndex]);

  std::vector<float> box_vec = cvtTensor(output_box);
  std::vector<float> score_vec = cvtTensor(output_score);
  std::vector<float> class_vec = cvtTensor(output_class);

  std::vector<size_t> result_id;
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
    std::vector<int> ids;
    cv::dnn::NMSBoxes(rects, scores, 0.6, 0.4, ids);
    for (int tmp : ids)
      cv::rectangle(image, rects[tmp], cv::Scalar(0, 255, 0), 3);
    cv::imwrite("./result.jpg", image);
  }
}