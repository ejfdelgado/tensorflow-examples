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

// ./cedula ./cedulas/img0000.jpg ../tensor_python/models -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -outfolder=./img0000/

using namespace cv;
using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

float extractVfromSegRes(std::vector<SegRes> lista, std::vector<std::string> classes)
{
  uint tamanio = lista.size();
  float total = 0;
  float conteo = 0;
  for (uint i = 0; i < tamanio; i++)
  {
    SegRes actual = lista[i];
    for (uint j = 0; j < classes.size(); j++)
    {
      std::string claseActual = classes[j];
      if (actual.c.compare(claseActual) == 0)
      {
        total += actual.v;
        conteo += 1;
      }
    }
  }
  float resultado = -1;
  if (conteo > 0)
  {
    resultado = total / classes.size();
  }
  std::cout << "conteo: " << conteo << ", total:" << total << ", resultado:" << resultado << std::endl;
  return resultado;
}

cv::Mat rotateImage(const cv::Mat &source, double angle)
{
  Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
  cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, angle, 1.0);
  cv::Mat dst;
  cv::warpAffine(source, dst, rot_mat, source.size());
  return dst;
}

float computeValueForRotation(
    std::vector<std::string> classesForEval,
    double degree,
    const cv::Mat &squared,
    std::vector<std::string> class_names,
    std::string modelPathString,
    std::string modeString,
    int normalize,
    float scoreThreshold,
    float sth,
    float nmsth,
    std::string outputFolder)
{
  cv::Mat image0;
  if (degree == 0)
  {
    image0 = squared.clone();
  }
  else
  {
    image0 = rotateImage(squared, degree);
  }
  std::vector<SegRes> myVector0 = runYoloOnce(
      class_names,
      modelPathString,
      modeString,
      image0,
      normalize,
      scoreThreshold,
      sth,
      nmsth,
      outputFolder);

  std::string myText = jsonifySegRes(myVector0);
  std::cout << myText << std::endl;

  float v0 = extractVfromSegRes(myVector0, classesForEval);
  return v0;
}

void computeHigherRotation(
    std::vector<std::string> class_names,
    std::vector<std::string> class_names2,
    std::string modelPathString,
    std::string modeString,
    const cv::Mat &image,
    int normalize,
    float scoreThreshold,
    float sth,
    float nmsth,
    std::string outfolder)
{
  std::string model = modelPathString + "/cedulas_vaale-fp16.tflite";
  std::string model2 = modelPathString + "/cedulas_vaale2-fp16.tflite";
  cv::Mat scaled;

  uint sizeScaled = 512;
  uint originalWidth = image.cols;
  uint originalHeight = image.rows;
  uint scaledWidth = sizeScaled;
  uint scaledHeight = sizeScaled;
  uint offsetX = 0;
  uint offsetY = 0;
  if (originalWidth > originalHeight)
  {
    scaledHeight = sizeScaled * image.rows / image.cols;
    offsetY = (sizeScaled - scaledHeight) * 0.5;
  }
  else
  {
    scaledWidth = sizeScaled * image.cols / image.rows;
    offsetX = (sizeScaled - scaledWidth) * 0.5;
  }
  cv::Mat squared(sizeScaled, sizeScaled, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::resize(image, scaled, cv::Size(scaledWidth, scaledHeight), 0, 0, cv::INTER_AREA);

  for (uint i = 0; i < scaledHeight; i++)
  {
    cv::Vec3b *ptr = scaled.ptr<cv::Vec3b>(i);
    cv::Vec3b *ptrDest = squared.ptr<cv::Vec3b>(i + offsetY);
    for (uint j = 0; j < scaledWidth; j++)
    {
      ptrDest[j + offsetX] = cv::Vec3b(ptr[j][0], ptr[j][1], ptr[j][2]);
    }
  }

  std::vector<std::string> classes4Eval2{"all"};

  std::vector<double> degrees;
  degrees.push_back(270);
  degrees.push_back(180);
  degrees.push_back(0);
  degrees.push_back(90);

  float maxValue = -1;
  float maxDegree = -1;
  for (uint i = 0; i < degrees.size(); i++)
  {
    double degree = degrees[i];
    float v = computeValueForRotation(
        classes4Eval2,
        degree,
        squared,
        class_names2,
        model2,
        modeString,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        "");
    std::cout << "degree:" << degree << ", val:" << v << std::endl;
    if (v > maxValue)
    {
      maxValue = v;
      maxDegree = degree;
    }
  }

  cv::Mat optimus90;
  if (maxDegree == 90)
  {
    cv::rotate(image, optimus90, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
  else if (maxDegree == 180)
  {
    cv::rotate(image, optimus90, cv::ROTATE_180);
  }
  else if (maxDegree == 270)
  {
    cv::rotate(image, optimus90, cv::ROTATE_90_CLOCKWISE);
  }
  else
  {
    optimus90 = image;
  }

  sizeScaled = 1024;
  scaledWidth = sizeScaled;
  scaledHeight = sizeScaled;
  originalWidth = optimus90.cols;
  originalHeight = optimus90.rows;
  if (originalWidth > originalHeight)
  {
    scaledHeight = sizeScaled * optimus90.rows / optimus90.cols;
  }
  else
  {
    scaledWidth = sizeScaled * optimus90.cols / optimus90.rows;
  }
  cv::resize(optimus90, scaled, cv::Size(scaledWidth, scaledHeight), 0, 0, cv::INTER_AREA);
  std::cout << "Optimus degree % 90: " << maxDegree << std::endl;

  std::vector<double> degreesDetail;
  degreesDetail.push_back(0);
  degreesDetail.push_back(5);
  degreesDetail.push_back(-5);
  degreesDetail.push_back(10);
  degreesDetail.push_back(-10);

  maxValue = -1;
  maxDegree = -1;
  std::vector<std::string> classes4Eval{"tit", "txt", "ph", "nom", "ape", "num", "sig", "bia", "fir", "e1"};
  for (uint i = 0; i < degreesDetail.size(); i++)
  {
    double degree = degreesDetail[i];
    std::stringstream ss;
    ss << outfolder << "deg" << degree << "-";

    float v = computeValueForRotation(
        classes4Eval,
        degree,
        scaled,
        class_names,
        model,
        modeString,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        ss.str());
    std::cout << "-----------------> degree:" << degree << ", val:" << v << std::endl;
    if (v > maxValue)
    {
      maxValue = v;
      maxDegree = degree;
    }
  }
  std::cout << "Optimus degree " << maxDegree << std::endl;

  cv::Mat finalImage;

  if (maxDegree == 0)
  {
    finalImage = optimus90;
  }
  else
  {
    finalImage = rotateImage(optimus90, maxDegree);
  }

  std::vector<SegRes> myVector = runYoloOnce(
      class_names,
      model,
      modeString,
      finalImage,
      normalize,
      scoreThreshold,
      sth,
      nmsth,
      outfolder);
  std::string myText = jsonifySegRes(myVector);
  std::cout << myText << std::endl;
}

int main(int argc, char *argv[])
{
  cv::CommandLineParser parser(argc, argv,
                               "{@image       |            |The image file to segment or classify}"
                               "{@model       |            |The folder for .tflite model files)}"
                               "{imageType it |IMREAD_COLOR|Can be: IMREAD_COLOR IMREAD_GRAYSCALE}"
                               "{mode       m |FLOAT       |Can be: FLOAT CHAR}"
                               "{normalize  n |10          |Can be: 11 10 256}"
                               "{threshold  th|0.6         |Threshold for score. Segmentation only.}"
                               "{sth          |0.5         |Threshold for class. Segmentation only.}"
                               "{nmsth        |0.45        |Threshold for nms. Segmentation only.}"
                               "{dpi          |70          |Dots per inch for text detection.}"
                               "{cedula       |            |Cedula.}"
                               "{cedtam       |850,550     |Cedula Tamanio W,H.}"
                               "{coords       |            |Coords.}"
                               "{ocrdir       |../mixed/   |Teseract folder dir where exists spa.traineddata.}"
                               "{outfolder    |            |The output folder, can be ./ Segmentation only.}");
  // parser.printMessage();
  String imagePathString = parser.get<String>("@image");
  String modelPathString = parser.get<String>("@model");
  String imageTypeString = parser.get<String>("imageType");
  String modeString = parser.get<String>("mode");
  int normalize = parser.get<int>("normalize");
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
  cv::Mat image;
  image = cv::imread(imagePathString.c_str(), imageType);
  if (!image.data)
  {
    printf("No image data \n");
    return EXIT_FAILURE;
  }

  std::vector<std::string> class_names;
  std::vector<std::string> class_names2;
  class_names = readLabelsFile((modelPathString + "/cedulas_vaale-fp16.txt").c_str());
  class_names2 = readLabelsFile((modelPathString + "/cedulas_vaale2-fp16.txt").c_str());

  // 1. Generar las rotaciones de +90 +180 +270 y mirar cuál da el score más alto

  computeHigherRotation(
      class_names,
      class_names2,
      modelPathString,
      modeString,
      image,
      normalize,
      scoreThreshold,
      sth,
      nmsth,
      outfolder);
  // std::string myText = jsonifySegRes(myVector);
  // std::cout << myText << std::endl;

  /*
  cv::Mat cedulaImage = cv::imread(cedula);
  if (cedulaImage.data)
  {
    std::vector<uint> cedtamvec = parseStringVector<uint>(cedtam);
    uint CEDULA_WIDTH = cedtamvec[0];
    uint CEDULA_HEIGHT = cedtamvec[1];
    postProcessCedula(cedulaImage, coords, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi, outfolder);
  }
  */

  return 0;
}
