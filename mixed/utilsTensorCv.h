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
auto explainTensor(TfLiteTensor *tensor) -> std::vector<T>;

auto explainTensor(std::string name, TfLiteTensor *tensor) -> void
{
  uint nelem = 1;
  uint tamanio = tensor->dims->size;
  std::cout << name << ".size=" << tamanio << "-----------------------------" << std::endl;
  for (int i = 0; i < tamanio; ++i)
  {
    uint dimension = tensor->dims->data[i];
    std::cout << name << "." << i << "=" << dimension << std::endl;
    nelem *= dimension;
  }
  std::cout << name << ".total=" << nelem << std::endl;
}

template <typename T>
void printTopClass(
    std::unique_ptr<tflite::Interpreter> *interpreter,
    uint classIndex,
    std::vector<std::string> class_names,
    float scoreThreshold)
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
void printSegmented(
    std::unique_ptr<tflite::Interpreter> *interpreter,
    float scoreThreshold,
    uint boxIndex,
    uint scoreIndex,
    uint classIndex,
    std::vector<std::string> class_names,
    std::string outfolder,
    cv::Mat image)
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

  if (outfolder.compare("") != 0)
  {
    std::vector<int> ids;
    cv::dnn::NMSBoxes(rects, scores, 0.6, 0.4, ids);
    for (int tmp : ids)
      cv::rectangle(image, rects[tmp], cv::Scalar(0, 255, 0), 3);
    std::string fullPath = outfolder + "result.jpg";
    cv::imwrite(fullPath.c_str(), image);
  }
}

// https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/
void printYoloV5(
    std::unique_ptr<tflite::Interpreter> *interpreter,
    cv::Mat &input_image,
    const std::vector<std::string> &class_names,
    float CONFIDENCE_THRESHOLD,
    float SCORE_THRESHOLD,
    float NMS_THRESHOLD,
    uint INPUT_WIDTH,
    uint INPUT_HEIGHT,
    std::string outfolder)
{
  const int THICKNESS = 1;

  // Colors.
  cv::Scalar BLACK = cv::Scalar(0, 0, 0);
  cv::Scalar BLUE = cv::Scalar(255, 178, 50);
  cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
  cv::Scalar RED = cv::Scalar(0, 0, 255);

  TfLiteTensor *output_box0 = (*interpreter)->tensor((*interpreter)->outputs()[0]);

  uint rows = output_box0->dims->data[1];
  uint dimensions = output_box0->dims->data[2];
  uint totalSize = rows * dimensions;
  uint numberOfClasses = dimensions - 5;

  // Initialize vectors to hold respective outputs while unwrapping detections.
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  // Resizing factor.
  /*
  float x_factor = input_image.cols / (float)INPUT_WIDTH;
  float y_factor = input_image.rows / (float)INPUT_HEIGHT;
  */
  float x_factor = input_image.cols;
  float y_factor = input_image.rows;
  float *data = (float *)output_box0->data.f;

  // 25200 for default size 640.
  // Iterate through 25200 detections.
  for (uint i = 0; i < rows; ++i)
  {
    // Center.
    float cx = data[0];
    float cy = data[1];
    // Box dimension.
    float w = data[2];
    float h = data[3];
    float confidence = data[4];

    // Discard bad detections and continue.
    if (confidence >= CONFIDENCE_THRESHOLD)
    {
      float *classes_scores = data + 5;
      // Create a 1x85 Mat and store class scores of 80 classes.
      cv::Mat scores(1, numberOfClasses, CV_32FC1, classes_scores);
      // Perform minMaxLoc and acquire the index of best class  score.
      cv::Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      // Continue if the class score is above the threshold.
      if (max_class_score > SCORE_THRESHOLD)
      {
        // Store class ID and confidence in the pre-defined respective vectors.
        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);
        // Bounding box coordinates.
        int left = int((cx - 0.5 * w) * x_factor);
        int top = int((cy - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);
        // Store good detections in the boxes vector.
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    // Jump to the next row.
    data += dimensions;
  }

  // Perform Non-Maximum Suppression and draw predictions.
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
  std::cout << "[";
  uint nFoundObjects = indices.size();
  uint classSize = class_names.size();
  for (uint i = 0; i < nFoundObjects; i++)
  {
    int idx = indices[i];
    cv::Rect box = boxes[idx];
    int left = box.x;
    int top = box.y;
    int width = box.width;
    int height = box.height;
    uint classIndex = class_ids[idx];
    // Draw bounding box.
    cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);
    std::cout << "{\"i\":" << classIndex << ", ";
    std::cout << "\"xi\":\"" << left << "\", ";
    std::cout << "\"xf\":\"" << left + width << "\", ";
    std::cout << "\"yi\":\"" << top << "\", ";
    std::cout << "\"yf\":\"" << top + height << "\", ";
    if (classIndex < classSize)
    {
      std::cout << "\"c\":\"" << class_names[classIndex] << "\", ";
    }
    std::cout << "\"v\":" << confidences[idx] << "}";
    if (i < nFoundObjects - 1)
    {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  if (outfolder.compare("") != 0)
  {
    std::string fullPath = outfolder + "result.jpg";
    cv::imwrite(fullPath.c_str(), input_image);
  }
}