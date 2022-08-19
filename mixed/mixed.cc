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
#include <tesseract/baseapi.h>

#include <regex>
#include <sstream>

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
// ./mixed ../tensor_python/models/cat.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./
// ./mixed ../tensor_python/models/bee.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./
// ./mixed ../tensor_python/models/zebra.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./

// ./mixed ../tensor_python/models/zebra.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./ -coords=1384,368,3528,332,1484,1608,3436,1584 -cedula=../tensor_python/models/cedula/001.jpg -dpi=200 -cedtam=950,650
// ./mixed ../tensor_python/models/zebra.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./ -coords=122,1840,1904,1884,138,3058,1902,2966 -cedula=../tensor_python/models/cedula/002.jpg -dpi=200 -cedtam=950,650
// ./mixed ../tensor_python/models/zebra.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./ -coords=1340,448,3456,420,1264,1796,3556,1780 -cedula=../tensor_python/models/cedula/003.jpg -dpi=200 -cedtam=950,650
// ./mixed ../tensor_python/models/zebra.jpg ../tensor_python/models/yolo/best-fp16.tflite -labels=../tensor_python/models/yolo/labels.txt -it=IMREAD_COLOR -m=FLOAT -n=10 -th=0.7 -yi=0 -outfolder=./ -coords=1228,388,3424,216,1288,1828,3652,1592 -cedula=../tensor_python/models/cedula/004.jpg -dpi=200 -cedtam=950,650

/*
sudo apt install tesseract-ocr libtesseract-dev
export TESSDATA_PREFIX=../mixed/
*/

using namespace cv;
using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

template <typename T>
std::vector<T> parseStringVector(std::string texto)
{
  const std::string espacios = std::regex_replace(texto, std::regex(","), " ");
  std::stringstream iss(espacios);

  T number;
  std::vector<T> myNumbers;
  while (iss >> number)
    myNumbers.push_back(number);
  return myNumbers;
}

std::string cleanText(std::string texto)
{
  const std::string clean1 = std::regex_replace(texto, std::regex("[^A-Za-z0-9ÁÉÍÓÚÜáéíóúü\\s]"), "");
  const std::string clean2 = std::regex_replace(clean1, std::regex("\\s+$"), "");
  return clean2;
}

std::vector<cv::Point2f> parseStringPoint2f(std::string texto)
{
  std::vector<int> numeros = parseStringVector<int>(texto);
  uint halfSize = numeros.size() / 2;

  std::vector<cv::Point2f> resultado;
  for (uint i = 0; i < halfSize; i++)
  {
    float x = numeros[2 * i];
    float y = numeros[2 * i + 1];
    resultado.push_back(cv::Point2f(x, y));
  }
  return resultado;
}

std::string extractText(cv::Mat dilate_dst, float xxi, float yyi, float xxf, float yyf, float CEDULA_WIDTH, float CEDULA_HEIGHT, std::string folderTrain, int dpi)
{
  // std::cout << xxi << ", " << yyi << ", " << xxf << ", " << yyf << std::endl;
  tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
  // https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
  ocr->Init(folderTrain.c_str(), "spa", tesseract::OEM_LSTM_ONLY);
  ocr->SetPageSegMode(tesseract::PSM_AUTO);
  ocr->SetImage(dilate_dst.data, dilate_dst.cols, dilate_dst.rows, dilate_dst.channels(), dilate_dst.step);

  uint xi = xxi * CEDULA_WIDTH;
  uint yi = yyi * CEDULA_HEIGHT;
  uint xf = xxf * CEDULA_WIDTH;
  uint yf = yyf * CEDULA_HEIGHT;

  ocr->SetRectangle(xi, yi, xf - xi, yf - yi);
  ocr->SetSourceResolution(dpi);

  ocr->Recognize(0);

  tesseract::ResultIterator *ri = ocr->GetIterator();
  tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

  std::string response = "";
  if (ri != 0)
  {
    do
    {
      const string word = ri->GetUTF8Text(level);
      response = response + word + " ";
    } while (ri->Next(level));
  }

  delete ocr;
  delete ri;

  return response;
}

void postProcessCedula(
    cv::Mat src,
    std::string coords,
    uint CEDULA_WIDTH,
    uint CEDULA_HEIGHT,
    std::string TRAINED_FOLDER,
    uint dpi,
    std::string outfolder)
{
  std::vector<cv::Point2f> sourcePoints = parseStringPoint2f(coords);
  cv::Mat dest = cutImage(src, sourcePoints, CEDULA_WIDTH, CEDULA_HEIGHT);

  // equalizar
  cv::Mat grayScale;
  cv::cvtColor(dest, grayScale, cv::COLOR_BGR2GRAY);
  // cv::Mat equalized;
  // cv::equalizeHist( grayScale, equalized );

  int erosion_type;
  int erosion_size = 1;
  // erosion_type = cv::MORPH_RECT;
  erosion_type = cv::MORPH_CROSS;
  // erosion_type = cv::MORPH_ELLIPSE;

  cv::Mat erosionElement = cv::getStructuringElement(erosion_type,
                                                     Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                     Point(erosion_size, erosion_size));
  cv::Mat erosion_dst;
  cv::erode(grayScale, erosion_dst, erosionElement);
  cv::Mat dilate_dst;
  cv::dilate(erosion_dst, dilate_dst, erosionElement);

  cv::imwrite("./ocr.jpg", dilate_dst);

  // 100.f/CEDULA_WIDTH, 100.f/CEDULA_HEIGHT, 100.f/CEDULA_WIDTH, 100.f/CEDULA_HEIGHT
  std::string apellidos = extractText(dilate_dst, 0.0105263, 0.363077, 0.581053, 0.447692, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi);
  apellidos = cleanText(apellidos);
  std::string nombres = extractText(dilate_dst, 0.0115789, 0.506154, 0.587368, 0.593846, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi);
  nombres = cleanText(nombres);
  std::string numCedula = extractText(dilate_dst, 0.152632, 0.289231, 0.589474, 0.38, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi);
  numCedula = cleanText(numCedula);

  std::string photoPath;
  std::string signaturePath;
  if (outfolder.compare("") == 0)
  {
    outfolder = "./";
  }
  photoPath = outfolder + "photo.jpg";
  signaturePath = outfolder + "signature.jpg";
  std::vector<cv::Point2f> sourcePointsPhoto = parseStringPoint2f("541,70,925,70,541,578,925,578");
  cv::Mat photoImage = cutImage(dest, sourcePointsPhoto, 350, 440);
  cv::imwrite(photoPath.c_str(), photoImage);

  std::vector<cv::Point2f> sourcePointsSign = parseStringPoint2f("45,428,515,428,45,586,515,586");
  cv::Mat signatureImage = cutImage(dest, sourcePointsSign, 515 - 45, 586 - 428);
  cv::imwrite(signaturePath.c_str(), signatureImage);

  std::cout << "{\"id\":\"";
  std::cout << numCedula;
  std::cout << "\", \"names\":\"";
  std::cout << nombres;
  std::cout << "\", \"lastnames\":\"";
  std::cout << apellidos;
  std::cout << "\", \"photoPath\":\"";
  std::cout << photoPath;
  std::cout << "\", \"signaturePath\":\"";
  std::cout << signaturePath;
  std::cout << "\"}" << std::endl;
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
                               "{yoloIndex yi |-1          |Number of yolo index. Segmentation only.}"
                               "{boxIndex   bi|-1          |Number of box index. Segmentation only.}"
                               "{threshold  th|0.6         |Threshold for score. Segmentation only.}"
                               "{sth          |0.5         |Threshold for class. Segmentation only.}"
                               "{nmsth        |0.45        |Threshold for nms. Segmentation only.}"
                               "{dpi          |70          |Dots per inch for text detection.}"
                               "{cedula       |            |Cedula.}"
                               "{cedtam       |850,550     |Cedula Tamanio W,H.}"
                               "{coords       |            |Coords.}"
                               "{ocrdir       |../mixed/   |Teseract folder dir.}"
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
  int yoloIndex = parser.get<int>("yoloIndex");
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

  if (yoloIndex >= 0)
  {
    printYoloV5(
        &interpreter,
        image,
        class_names,
        scoreThreshold, // CONFIDENCE_THRESHOLD=0.45
        sth,            // SCORE_THRESHOLD -sth para la clase
        nmsth,          // NMS_THRESHOLD -nmsth
        WIDTH_M,
        HEIGHT_M,
        outfolder);
  }

  std::vector<uint> cedtamvec = parseStringVector<uint>(cedtam);
  uint CEDULA_WIDTH = cedtamvec[0];
  uint CEDULA_HEIGHT = cedtamvec[1];
  cv::Mat cedulaImage = cv::imread(cedula);

  postProcessCedula(cedulaImage, coords, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi, outfolder);

  return 0;
}
