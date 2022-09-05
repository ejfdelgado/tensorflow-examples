#include <string>
#include <iostream>
#include <fstream>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

// TODO
// OCR de números propio
// Si en las rotaciones ninguno tiene un minimos score de 70 se termina
// Recortar también los nombres y apellidos

std::string extractText(cv::Mat dilate_dst, float xxi, float yyi, float xxf, float yyf, float CEDULA_WIDTH, float CEDULA_HEIGHT, std::string folderTrain, int dpi, float UMBRAL, bool isNumber)
{
    // std::cout << xxi << ", " << yyi << ", " << xxf << ", " << yyf << std::endl;
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    // https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
    ocr->Init(folderTrain.c_str(), "spa", tesseract::OEM_LSTM_ONLY);
    //ocr->SetPageSegMode(tesseract::PSM_AUTO);
    ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr->SetImage(dilate_dst.data, dilate_dst.cols, dilate_dst.rows, dilate_dst.channels(), dilate_dst.step);

    if (isNumber) {
        ocr->SetVariable("tessedit_char_blacklist", "!?@#$%&*()<>_-+=/:;'\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        ocr->SetVariable("tessedit_char_whitelist", "0123456789");
        ocr->SetVariable("classify_bln_numeric_mode", "1");
    } else {
        ocr->SetVariable("tessedit_char_blacklist", "0123456789!?@#$%&*()<>_-+=/:;'\"abcdefghijklmnopqrstuvwxyz");
        ocr->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ ");
    }

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

bool firstSegResLeft(const SegRes &a, const SegRes &b)
{
  return b.cx > a.cx;
}

void postProcessCedula(
    cv::Mat src,
    std::vector<cv::Point2f> sourcePoints,
    uint CEDULA_WIDTH,
    uint CEDULA_HEIGHT,
    std::string TRAINED_FOLDER,
    uint dpi,
    std::string outfolder,
    std::string imageIdentifier,
    std::vector<std::string> class_names,
    std::string modelPathString,
    std::string modeString,
    int normalize,
    float scoreThreshold,
    float sth,
    float nmsth
    )
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

    cv::Mat dilate_dst = closing(dest, 1);
    cv::imwrite(ocrPath.c_str(), dilate_dst);

    float ROI_ID_X1 = 116;
    float ROI_ID_Y1 = 130;
    float ROI_ID_X2 = 501;
    float ROI_ID_Y2 = 214;
    float CONFIDENCE_ID = 96;

    // 100.f/CEDULA_WIDTH, 100.f/CEDULA_HEIGHT, 100.f/CEDULA_WIDTH, 100.f/CEDULA_HEIGHT
    std::string apellidos = extractText(dilate_dst, 0.0105263, 0.363077, 0.581053, 0.447692, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi, 90, false);
    apellidos = cleanText(apellidos);
    std::string nombres = extractText(dilate_dst, 0.0115789, 0.506154, 0.587368, 0.593846, CEDULA_WIDTH, CEDULA_HEIGHT, TRAINED_FOLDER, dpi, 90, false);
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
        CONFIDENCE_ID,
        true);
    numCedula = cleanNumber(numCedula);

    cv::Rect myROI(ROI_ID_X1, ROI_ID_Y1, ROI_ID_X2 - ROI_ID_X1, ROI_ID_Y2 - ROI_ID_Y1);
    cv::Mat roiId = dest(myROI);

    uint offsetXOut;
    uint offsetYOut;
    cv::Mat roiIdSquare = squareImage(roiId, 640, &offsetXOut, &offsetYOut);
    std::stringstream num_final;
    std::vector<SegRes> ocrNumbers = runYoloOnce(
        class_names,
        modelPathString,
        modeString,
        roiIdSquare,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        outfolder);
    uint num_total = ocrNumbers.size();
    std::sort(ocrNumbers.begin(), ocrNumbers.end(), firstSegResLeft);
    std::cout << "numeros: " << num_total << std::endl;
    float num_avg_w = 0;
    float num_avg_h = 0;
    for (int k=0; k<num_total; k++) {
        SegRes temp = ocrNumbers[k];
        num_avg_w += temp.xf - temp.xi;
        num_avg_h += temp.yf - temp.yi;
    }
    num_avg_w = num_avg_w / ((float)num_total);
    num_avg_h = num_avg_h / ((float)num_total);
    std::cout << "avgs: " << num_avg_w << " x " << num_avg_h << std::endl;

    // Elimino los que están por fuera del rango promedio
    float UMBRAL_ALTURA_MINIMA = 0.8;
    float UMBRAL_ALTURA_MAXIMA = 1.2;
    float UMBRAL_GAP = 0.62;
    float UMBRAL_CLOSE_Y = 0.075;
    float num_comp_w;
    float num_comp_h;
    for (uint k=0; k<ocrNumbers.size(); k++) {
        SegRes temp = ocrNumbers[k];
        num_comp_h = (temp.yf - temp.yi)/num_avg_h;
        if (num_comp_h < UMBRAL_ALTURA_MINIMA || num_comp_h > UMBRAL_ALTURA_MAXIMA) {
            std::cout << "borrando " << temp.c << ", h: " << num_comp_h << std::endl;
            ocrNumbers.erase(ocrNumbers.begin() + k);
        }
    }
    num_total = ocrNumbers.size();

    // altura promedio.
    // ancho promedio.
    // Calculo gap entre el actual y el anterior
    float lastXF = -1;
    float gapBetween = 0;
    uint ocrWidth = roiIdSquare.cols;
    uint ocrHeight = roiIdSquare.rows;
    float yUpProportion;
    float yDownProportion;
    bool riskyNumber = false;
    for (int k=0; k<num_total; k++) {
        SegRes temp = ocrNumbers[k];
        num_comp_h = (temp.yf - temp.yi)/num_avg_h;
        num_comp_w = (temp.xf - temp.xi)/num_avg_w;
        //Calculo gap
        if (lastXF >= 0) {
            gapBetween = (temp.xi - lastXF)/num_avg_w;
        }
        lastXF = temp.xf;
        if (gapBetween > UMBRAL_GAP) {
            num_final << "_";
        }
        yUpProportion = (temp.yi-offsetYOut)/num_avg_h;
        yDownProportion = (ocrHeight - offsetYOut - temp.yf)/num_avg_h;

        if (yUpProportion < UMBRAL_CLOSE_Y || yDownProportion < UMBRAL_CLOSE_Y) {
            riskyNumber = true;
            break;
        }

        num_final << temp.c;
        
        std::cout << temp.c << ", gap=" << gapBetween << ", " << num_comp_w << " x " << num_comp_h << std::endl;
        std::cout << "too close to top/bottom border: " << yUpProportion << ", " << yDownProportion << std::endl;
    }
    std::cout << std::endl;

    std::string myOCR = "";
    if (!riskyNumber) {
        myOCR = num_final.str();
    }

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

    myfile << "{\n\t\"old_id\":\"";
    myfile << numCedula;
    myfile << "\",\n\t\"id\":\"";
    myfile << myOCR;
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