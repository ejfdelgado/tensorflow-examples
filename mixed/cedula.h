#include <string>
#include <iostream>
#include <fstream>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

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
    std::vector<cv::Point2f> sourcePoints,
    uint CEDULA_WIDTH,
    uint CEDULA_HEIGHT,
    std::string TRAINED_FOLDER,
    uint dpi,
    std::string outfolder)
{
    // std::vector<cv::Point2f> sourcePoints = parseStringPoint2f(coords);
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
    std::string normalizedImage;
    std::string jsonPath;
    if (outfolder.compare("") == 0)
    {
        outfolder = "./";
    }
    photoPath = outfolder + "photo.jpg";
    signaturePath = outfolder + "signature.jpg";
    normalizedImage = outfolder + "normalized.jpg";
    jsonPath = outfolder + "response.json";
    std::vector<cv::Point2f> sourcePointsPhoto;
    sourcePointsPhoto.push_back(cv::Point2f(CEDULA_WIDTH * 541.f / 950.f, CEDULA_HEIGHT * 70.f / 650.f));
    sourcePointsPhoto.push_back(cv::Point2f(CEDULA_WIDTH * 925.f / 950.f, CEDULA_HEIGHT * 70.f / 650.f));
    sourcePointsPhoto.push_back(cv::Point2f(CEDULA_WIDTH * 541.f / 950.f, CEDULA_HEIGHT * 578.f / 650.f));
    sourcePointsPhoto.push_back(cv::Point2f(CEDULA_WIDTH * 925.f / 950.f, CEDULA_HEIGHT * 578.f / 650.f));
    cv::Mat photoImage = cutImage(dest, sourcePointsPhoto, 350, 440);
    cv::imwrite(photoPath.c_str(), photoImage);

    std::vector<cv::Point2f> sourcePointsSign;
    sourcePointsSign.push_back(cv::Point2f(CEDULA_WIDTH * 45.f / 950.f, CEDULA_HEIGHT * 428.f / 650.f));
    sourcePointsSign.push_back(cv::Point2f(CEDULA_WIDTH * 515.f / 950.f, CEDULA_HEIGHT * 428.f / 650.f));
    sourcePointsSign.push_back(cv::Point2f(CEDULA_WIDTH * 45.f / 950.f, CEDULA_HEIGHT * 586.f / 650.f));
    sourcePointsSign.push_back(cv::Point2f(CEDULA_WIDTH * 515.f / 950.f, CEDULA_HEIGHT * 586.f / 650.f));
    cv::Mat signatureImage = cutImage(dest, sourcePointsSign, 470, 158);
    cv::imwrite(signaturePath.c_str(), signatureImage);
    cv::imwrite(normalizedImage.c_str(), dest);

    std::ofstream myfile;
    myfile.open(jsonPath.c_str(), std::ios::trunc);

    myfile << "{\"id\":\"";
    myfile << numCedula;
    myfile << "\", \"names\":\"";
    myfile << nombres;
    myfile << "\", \"lastnames\":\"";
    myfile << apellidos;
    myfile << "\", \"photoPath\":\"";
    myfile << photoPath;
    myfile << "\", \"signaturePath\":\"";
    myfile << signaturePath;
    myfile << "\"}" << std::endl;

    myfile.close();
}