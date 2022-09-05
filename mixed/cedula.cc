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
#include "solvePnP.h"

// cmake ../mixed
// cmake --build . -j 4

// ./cedula ./cedulas/img0002.jpg ../tensor_python/models -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -outfolder=./process/img0002/

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

Data3D getModel3D(std::string clase)
{
  if (clase.compare("tit") == 0)
  {
    return {320, 108, 0};
  }
  else if (clase.compare("num") == 0)
  {
    return {95, 212, 0};
  }
  else if (clase.compare("ape") == 0)
  {
    return {107, 304, 0};
  }
  else if (clase.compare("nom") == 0)
  {
    return {102, 401, 0};
  }
  else if (clase.compare("e1") == 0)
  {
    return {75, 78, 0};
  }
  else if (clase.compare("bia") == 0)
  {
    return {509, 93, 0};
  }
  else if (clase.compare("ph") == 0)
  {
    return {767, 306, 0};
  }
  else if (clase.compare("fir") == 0)
  {
    return {291, 532, 0};
  }
  else if (clase.compare("sig") == 0)
  {
    return {165, 575, 0};
  }
  else if (clase.compare("e4") == 0)
  {
    return {1004, 626, 0};
  }
  return {-1, -1, -1};
}

std::vector<cv::Point2f> estimateCorners(std::vector<SegRes> myVector)
{
  uint nFoundObjects = myVector.size();
  std::vector<Data2D> ref2D;
  std::vector<Data3D> ref3D;
  std::vector<Data3D> question = {{0, 0, 0}, {1004, 0, 0}, {0, 626, 0}, {1004, 626, 0}};
  for (uint i = 0; i < nFoundObjects; i++)
  {
    SegRes temp = myVector[i];
    Data3D oneRef3D = getModel3D(temp.c);
    if (oneRef3D.x >= 0)
    {
      ref2D.push_back({temp.cx, temp.cy});
      ref3D.push_back(oneRef3D);
    }
  }
  std::vector<cv::Point2f> response;
  if (ref2D.size() >= 3)
  {
    response = guessPoints(question, ref2D, ref3D);
    for (unsigned int i = 0; i < response.size(); ++i)
    {
      if (response[i].x < 0)
      {
        response[i].x = 0;
      }
      if (response[i].y < 0)
      {
        response[i].y = 0;
      }
      std::cout << "Projected to " << response[i] << std::endl;
    }
  }
  return response;
}

void computeHigherRotation(
    std::vector<std::string> class_names,
    std::vector<std::string> class_names2,
    std::vector<std::string> class_names3,
    std::string modelPathString,
    std::string modeString,
    cv::Mat &image,
    int normalize,
    float scoreThreshold,
    float sth,
    float nmsth,
    int dpi,
    std::string outfolder,
    std::string TRAINED_FOLDER,
    std::string imageIdentifier,
    float MIN_UMBRAL_CEDULA)
{
  std::string model = modelPathString + "/cedulas_vaale-fp16.tflite";
  std::string model2 = modelPathString + "/cedulas_vaale2-fp16.tflite";
  std::string model3 = modelPathString + "/roi_ids1-fp16.tflite";

  uint sizeScaled = 512;

  cv::Mat squared = squareImage(image, sizeScaled);

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

  if (maxValue < MIN_UMBRAL_CEDULA) {
    std::ofstream myfile;
    std::string jsonPath;
    jsonPath = outfolder + "actual.json";
    myfile.open(jsonPath.c_str(), std::ios::trunc);

    myfile << "{\n\t\"umbral\":\"";
    myfile << maxValue;
    myfile << "\"\n}" << std::endl;

    myfile.close();
    return;
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
  uint scaledWidth = sizeScaled;
  uint scaledHeight = sizeScaled;
  uint originalWidth = optimus90.cols;
  uint originalHeight = optimus90.rows;
  if (originalWidth > originalHeight)
  {
    scaledHeight = sizeScaled * optimus90.rows / optimus90.cols;
  }
  else
  {
    scaledWidth = sizeScaled * optimus90.cols / optimus90.rows;
  }
  cv::Mat scaled;
  cv::resize(optimus90, scaled, cv::Size(scaledWidth, scaledHeight), 0, 0, cv::INTER_AREA);
  std::cout << "Optimus degree % 90: " << maxDegree << std::endl;

  cv::Mat finalImage;

  std::vector<SegRes> myVector = runYoloOnce(
      class_names,
      model,
      modeString,
      optimus90,
      normalize,
      scoreThreshold,
      sth,
      nmsth,
      "");
  std::string myText = jsonifySegRes(myVector);
  std::cout << myText << std::endl;

  // Debo calcular el pnp
  std::vector<cv::Point2f> coords = estimateCorners(myVector);
  if (coords.size() >= 4)
  {
    uint CEDULA_WIDTH = 850;
    uint CEDULA_HEIGHT = 550;
    postProcessCedula(
      optimus90, 
      coords, 
      CEDULA_WIDTH, 
      CEDULA_HEIGHT, 
      TRAINED_FOLDER, 
      dpi, 
      outfolder, 
      imageIdentifier, 
      class_names3, 
      model3,
      modeString,
      normalize,
      scoreThreshold,
      sth,
      nmsth
      );
  }
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
                               "{escedula     |0.8         |Es una cedula.}"
                               "{dpi          |200         |Dots per inch for text detection.}"
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
  float MIN_UMBRAL_CEDULA = parser.get<float>("escedula");
  String outfolder = parser.get<String>("outfolder");
  String cedula = parser.get<String>("cedula");
  String cedtam = parser.get<String>("cedtam");
  String coords = parser.get<String>("coords");
  std::string TRAINED_FOLDER = parser.get<String>("ocrdir");

  std::string soloNombre = getRegexGroup("([^/]+)$", imagePathString, 1);
  soloNombre = getRegexGroup("([^.]+)", soloNombre, 1);
  std::cout << "soloNombre: " << soloNombre << std::endl;

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
  std::vector<std::string> class_names3;
  class_names = readLabelsFile((modelPathString + "/cedulas_vaale-fp16.txt").c_str());
  class_names2 = readLabelsFile((modelPathString + "/cedulas_vaale2-fp16.txt").c_str());
  class_names3 = readLabelsFile((modelPathString + "/roi_ids-fp16.txt").c_str());

  // 1. Generar las rotaciones de +90 +180 +270 y mirar cuál da el score más alto

  computeHigherRotation(
      class_names,
      class_names2,
      class_names3,
      modelPathString,
      modeString,
      image,
      normalize,
      scoreThreshold,
      sth,
      nmsth,
      dpi,
      outfolder,
      TRAINED_FOLDER,
      soloNombre,
      MIN_UMBRAL_CEDULA);

  // std::string myText = jsonifySegRes(myVector);
  // std::cout << myText << std::endl;

  return 0;
}
