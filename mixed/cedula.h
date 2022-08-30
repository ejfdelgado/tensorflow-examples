#include <string>
#include <iostream>
#include <fstream>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

std::string extractText(cv::Mat dilate_dst, float xxi, float yyi, float xxf, float yyf, float CEDULA_WIDTH, float CEDULA_HEIGHT, std::string folderTrain, int dpi, float UMBRAL)
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
            char *ctext = ri->GetUTF8Text(level);
            if (ctext != NULL)
            {
                float conf = ri->Confidence(tesseract::RIL_SYMBOL);
                const std::string word = ctext;
                std::cout << "word: " << word << ", conf: " << conf << std::endl;
                if (conf > UMBRAL)
                {
                    response = response + word + " ";
                }
            }
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
    std::string outfolder,
    std::string imageIdentifier)
{
    std::string photoPath;
    std::string signaturePath;
    std::string normalizedImage;
    std::string jsonPath;
    std::string ocrPath;
    std::string roiIdPath;
    if (outfolder.compare("") == 0)
    {
        outfolder = "./";
    }
    photoPath = outfolder + "photo.jpg";
    signaturePath = outfolder + "signature.jpg";
    normalizedImage = outfolder + "normalized.jpg";
    jsonPath = outfolder + "actual.json";
    ocrPath = outfolder + "ocr.jpg";
    roiIdPath = outfolder + "roi_id_" + imageIdentifier + ".jpg";
    // std::vector<cv::Point2f> sourcePoints = parseStringPoint2f(coords);
    cv::Mat dest = cutImage(src, sourcePoints, CEDULA_WIDTH, CEDULA_HEIGHT);
    cv::imwrite(normalizedImage.c_str(), dest);

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

    cv::imwrite(ocrPath.c_str(), dilate_dst);

    float ROI_ID_X1 = 116;
    float ROI_ID_Y1 = 130;
    float ROI_ID_X2 = 501;
    float ROI_ID_Y2 = 214;
    float CONFIDENCE_ID = 96;

    // 100.f/CEDULA_WIDTH, 100.f/CEDULA_HEIGHT, 100.f/CEDULA_WIDTH, 100.f/CEDULA_HEIGHT
    std::string apellidos = extractText(dilate_dst, 0.0105263, 0.363077, 0.581053, 0.447692, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi, 90);
    apellidos = cleanText(apellidos);
    std::string nombres = extractText(dilate_dst, 0.0115789, 0.506154, 0.587368, 0.593846, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi, 90);
    nombres = cleanText(nombres);
    std::string numCedula = extractText(dilate_dst, 
        ROI_ID_X1 / CEDULA_WIDTH, 
        ROI_ID_Y1 / CEDULA_HEIGHT, 
        ROI_ID_X2 / CEDULA_WIDTH, 
        ROI_ID_Y2 / CEDULA_HEIGHT, 
        CEDULA_WIDTH, 
        CEDULA_HEIGHT, 
        TRAINED_FOLDER, 
        dpi, 
        CONFIDENCE_ID);
    numCedula = cleanNumber(numCedula);

    cv::Rect myROI(ROI_ID_X1, ROI_ID_Y1, ROI_ID_X2 - ROI_ID_X1, ROI_ID_Y2 - ROI_ID_Y1);
    cv::Mat roiId = dest(myROI);
    cv::imwrite(roiIdPath.c_str(), roiId);

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

    std::ofstream myfile;
    myfile.open(jsonPath.c_str(), std::ios::trunc);

    myfile << "{\n\t\"id\":\"";
    myfile << numCedula;
    myfile << "\",\n\t\"names\":\"";
    myfile << nombres;
    myfile << "\",\n\t\"lastnames\":\"";
    myfile << apellidos;
    myfile << "\",\n\t\"photoPath\":\"";
    myfile << photoPath;
    myfile << "\",\n\t\"signaturePath\":\"";
    myfile << signaturePath;
    myfile << "\"\n}" << std::endl;

    myfile.close();
}