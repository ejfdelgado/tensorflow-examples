#include <regex>
#include <sstream>
#include <string>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>

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
            const std::string word = ri->GetUTF8Text(level);
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
                                                       cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                       cv::Point(erosion_size, erosion_size));
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