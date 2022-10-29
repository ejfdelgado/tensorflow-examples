#include <string>
#include <iostream>
#include <fstream>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <math.h>

// TODO
// OCR de números propio
// Si en las rotaciones ninguno tiene un minimos score de 70 se termina
// Recortar también los nombres y apellidos

std::string extractText(cv::Mat dilate_dst, float* theROI, float CEDULA_WIDTH, float CEDULA_HEIGHT, std::string folderTrain, int dpi, float UMBRAL, bool isNumber, std::string roiPath)
{
    float xxi = theROI[0] / CEDULA_WIDTH;
    float yyi = theROI[1] / CEDULA_HEIGHT;
    float xxf = theROI[2] / CEDULA_WIDTH;
    float yyf = theROI[3] / CEDULA_HEIGHT;
    // std::cout << xxi << ", " << yyi << ", " << xxf << ", " << yyf << std::endl;
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    // https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
    ocr->Init(folderTrain.c_str(), "spa", tesseract::OEM_LSTM_ONLY);
    ocr->SetImage(dilate_dst.data, dilate_dst.cols, dilate_dst.rows, dilate_dst.channels(), dilate_dst.step);
    ocr->SetPageSegMode(tesseract::PSM_AUTO);
    
    if (isNumber) {
        
        ocr->SetVariable("tessedit_char_blacklist", "!?@#$%&*()<>_-+=/:;'\"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        ocr->SetVariable("tessedit_char_whitelist", "0123456789");
        ocr->SetVariable("classify_bln_numeric_mode", "1");
    } else {
        //ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
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

    if (roiPath.length() > 0) {
        cv::Mat cropped_image = dilate_dst(cv::Range(yi,yf), cv::Range(xi,xf));
        cv::imwrite(roiPath.c_str(), cropped_image);
    }

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

std::string ocrFunNum(
    cv::Mat dest, 
    float ROI_ID_X1, 
    float ROI_ID_Y1, 
    float ROI_ID_X2, 
    float ROI_ID_Y2, 
    std::vector<std::string> class_names, 
    std::string modelPathString,
    std::string modeString,
    int normalize,
    float scoreThreshold,
    float sth,
    float nmsth,
    std::string outfolder
    ) {
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
    float num_med_cx = -1;
    float num_med_cy = -1;
    std::vector<float> num_med_cx_arr;
    std::vector<float> num_med_cy_arr;
    for (int k=0; k<num_total; k++) {
        SegRes temp = ocrNumbers[k];
        num_avg_w += temp.xf - temp.xi;
        num_avg_h += temp.yf - temp.yi;
        num_med_cx_arr.push_back(temp.cx);
        num_med_cy_arr.push_back(temp.cy);
    }
    // ordeno los vectores
    std::sort(num_med_cx_arr.begin(), num_med_cx_arr.end());
    std::sort(num_med_cy_arr.begin(), num_med_cy_arr.end());
    // calculo la mitad
    uint ix_mitad_x = ceil(((float)num_med_cx_arr.size())/2.0f);
    uint ix_mitad_y = ceil(((float)num_med_cy_arr.size())/2.0f);
    num_med_cx = num_med_cx_arr[ix_mitad_x];
    num_med_cy = num_med_cy_arr[ix_mitad_y];

    num_avg_w = num_avg_w / ((float)num_total);
    num_avg_h = num_avg_h / ((float)num_total);
    std::cout << "avgs: " << num_avg_w << " x " << num_avg_h << std::endl;
    std::cout << "median center: " << num_med_cx << " x " << num_med_cy << std::endl;

    // Elimino los que están por fuera del rango promedio
    float UMBRAL_ALTURA_MINIMA = 0.8;
    float UMBRAL_ALTURA_MAXIMA = 1.2;
    float UMBRAL_ANCHO_MINIMO = 0.3;
    float UMBRAL_ANCHO_MAXIMO = 1.5;
    float UMBRAL_CENTER_Y = 1.0;
    float UMBRAL_GAP = 0.62;
    float UMBRAL_CLOSE_Y = 0.075;
    float MIN_CONFIDENCE = 0.76;// Tal vez subir a 0.8
    float MIN_TAMANIO = 0.3;// Debe ser mayor a 0.27 y se prefiere pequeño
    float num_comp_w;
    float num_comp_h;
    float num_comp_cx;
    float num_comp_cy;
    float numTamanio;
    for (uint k=0; k<ocrNumbers.size(); k++) {
        SegRes temp = ocrNumbers[k];
        float ancho = (temp.xf - temp.xi);
        float alto = (temp.yf - temp.yi);
        numTamanio = ancho*alto/(num_avg_w*num_avg_h);
        num_comp_w = ancho/num_avg_w;
        num_comp_h = alto/num_avg_h;
        num_comp_cx = (temp.cx - num_med_cx)/num_avg_w;
        num_comp_cy = fabs(temp.cy - num_med_cy)/num_avg_h;
        std::cout << "?" << temp.c << "@" << temp.v << "x" << numTamanio << ", w: " << num_comp_w << ", h: " << num_comp_h << " cy: " << num_comp_cy << std::endl;
        if (
            num_comp_w < UMBRAL_ANCHO_MINIMO || 
            num_comp_w > UMBRAL_ANCHO_MAXIMO || 
            num_comp_h < UMBRAL_ALTURA_MINIMA || 
            num_comp_h > UMBRAL_ALTURA_MAXIMA ||
            num_comp_cy > UMBRAL_CENTER_Y ||
            temp.v < MIN_CONFIDENCE ||
            numTamanio < MIN_TAMANIO
        ) {
            std::cout << "Borrando " << temp.c << ", w: " << num_comp_w << ", h: " << num_comp_h << " cy: " << num_comp_cy << std::endl;
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
    float riskyNumberCount = 0;
    float riskyNumberTotal = 0;
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
            riskyNumberCount++;
        } else {
            riskyNumber = false;
        }
        riskyNumberTotal++;

        num_final << temp.c;
        
        std::cout << temp.c << ", gap=" << gapBetween << ", " << num_comp_w << " x " << num_comp_h << std::endl;
        if (riskyNumber) {
            std::cout << "too close to top/bottom border: [" << yUpProportion << ", " << yDownProportion << "] vs " << UMBRAL_CLOSE_Y << std::endl;
        }
    }
    std::cout << std::endl;

    float riskyScore = riskyNumberCount/riskyNumberTotal;
    float RISKY_THRESHOLD = (2.0f/10.0f);
    riskyNumber = (riskyScore > RISKY_THRESHOLD);

    if (riskyNumber) {
        std::cout << "Is Risky Score!: " << riskyScore << " vs " << RISKY_THRESHOLD << std::endl;
    } else {
        std::cout << "not risky score: " << riskyScore << " vs " << RISKY_THRESHOLD << std::endl;
    }

    std::string myOCR = "";
    if (!riskyNumber) {
        myOCR = num_final.str();
    }
    return myOCR;
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
    std::vector<std::string> class_names4,
    std::string modelPathString4,
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
    std::string roiNamePath;
    std::string roiLastNamePath;
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
    roiNamePath = outfolder + "roi_name01_" + imageIdentifier + ".jpg";
    roiLastNamePath = outfolder + "roi_name02_" + imageIdentifier + ".jpg";
    // std::vector<cv::Point2f> sourcePoints = parseStringPoint2f(coords);
    cv::Mat dest = cutImage(src, sourcePoints, CEDULA_WIDTH, CEDULA_HEIGHT);
    cv::imwrite(normalizedImage.c_str(), dest);

    cv::Mat dilate_dst = closing(dest, 1);
    //cv::imwrite(ocrPath.c_str(), dilate_dst);

    // Xi, Yi, Xf, Yf...
    float ROI_ID[4] = {116, 128, 501, 222};//{116, 128, 501, 214}
    float CONFIDENCE_ID = 96;

    float gap = 0.85;
    float WIDTH_NAMES = 502 * gap;
    float ROI_LAST_NAME[4] = {0, 177, WIDTH_NAMES, 268};
    float ROI_NAME[4] = {0, 267, WIDTH_NAMES, 348};
    
    std::string apellidos = extractText(dest, 
        ROI_LAST_NAME,
        CEDULA_WIDTH, 
        CEDULA_HEIGHT, 
        TRAINED_FOLDER, 
        dpi, 90, false, roiNamePath);
    apellidos = cleanText(apellidos);

    std::string nombres = extractText(dest, 
        ROI_NAME,
        CEDULA_WIDTH, 
        CEDULA_HEIGHT, 
        TRAINED_FOLDER, dpi, 90, false, roiLastNamePath);
    nombres = cleanText(nombres);

    std::string numCedula = extractText(dilate_dst, 
        ROI_ID,
        CEDULA_WIDTH, 
        CEDULA_HEIGHT, 
        TRAINED_FOLDER, 
        dpi, 
        CONFIDENCE_ID,
        true,
        "");
    numCedula = cleanNumber(numCedula);
    
    std::string myOCR = ocrFunNum(
        dest, 
        ROI_ID[0], 
        ROI_ID[1], 
        ROI_ID[2], 
        ROI_ID[3], 
        class_names, 
        modelPathString,
        modeString,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        outfolder
    );

    //cv::imwrite(roiIdPath.c_str(), roiId);

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
    myfile << "\",\n\t\"normalized\":\"";
    myfile << normalizedImage;
    myfile << "\"\n}" << std::endl;

    myfile.close();
}