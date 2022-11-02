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

void computeConnectedCharacters(
    std::vector<SegRes>& ocrCharaters, 
    std::vector<SegRes>& letrasConectadas, 
    float cx, 
    float cy, 
    float maxDistanceX,
    float maxDistanceY
    ) {
        if (ocrCharaters.size() == 0) { return; }
        uint k = 0;
        do {
            SegRes temp = ocrCharaters[k];
            float distanceX = abs(temp.cx - cx);
            float distanceY = abs(temp.cy - cy);
            if (distanceX < maxDistanceX && distanceY < maxDistanceY) {
                temp.connected = true;
                letrasConectadas.push_back(temp);
                ocrCharaters.erase(ocrCharaters.begin() + k);
            } else {
                k++;
            }
        } while (k < ocrCharaters.size());
        //std::cout << "Conectadas: ";
        for (uint k=0; k<letrasConectadas.size(); k++) {
            SegRes temp = letrasConectadas[k];
            //std::cout << temp.c;
        }
        //std::cout << std::endl;
}

std::string ocrFunTxt(
    cv::Mat dest, 
    float* ROI_ID,
    std::vector<std::string> class_names, 
    std::string modelPathString,
    std::string modeString,
    int normalize,
    float scoreThreshold,
    float sth,
    float nmsth,
    std::string outfolder
    ) {
    float ROI_ID_X1 = ROI_ID[0];
    float ROI_ID_Y1 = ROI_ID[1];
    float ROI_ID_X2 = ROI_ID[2];
    float ROI_ID_Y2 = ROI_ID[3];
    cv::Rect myROI(ROI_ID_X1, ROI_ID_Y1, ROI_ID_X2 - ROI_ID_X1, ROI_ID_Y2 - ROI_ID_Y1);
    cv::Mat roiId = dest(myROI);

    uint offsetXOut;
    uint offsetYOut;
    cv::Mat roiIdSquare = squareImage(roiId, 640, &offsetXOut, &offsetYOut);
    std::vector<SegRes> ocrCharaters = runYoloOnce(
        class_names,
        modelPathString,
        modeString,
        roiIdSquare,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        outfolder);
    uint letras_total = ocrCharaters.size();
    std::sort(ocrCharaters.begin(), ocrCharaters.end(), firstSegResLeft);
    std::cout << "Letras:";
    // 1.1. Buscar la mediana en X y Y de los centros, con esto tengo el centro de la frase
    float num_avg_w = 0;
    float num_avg_h = 0;
    float num_med_cx = -1;
    float num_med_cy = -1;
    float num_med_w = -1;
    float num_med_h = -1;
    std::vector<float> num_med_cx_arr;
    std::vector<float> num_med_cy_arr;
    std::vector<float> num_med_w_arr;
    std::vector<float> num_med_h_arr;
    for (uint k=0; k<letras_total; k++) {
        SegRes temp = ocrCharaters[k];
        std::string letra = temp.c;
        std::cout << letra;
        float w = temp.xf - temp.xi;
        float h = temp.yf - temp.yi;
        num_avg_w += w;
        num_avg_h += h;
        num_med_cx_arr.push_back(temp.cx);
        num_med_cy_arr.push_back(temp.cy);
        num_med_w_arr.push_back(w);
        num_med_h_arr.push_back(h);
    }
    std::cout << std::endl;
    // ordeno los vectores
    std::sort(num_med_cx_arr.begin(), num_med_cx_arr.end());
    std::sort(num_med_cy_arr.begin(), num_med_cy_arr.end());
    std::sort(num_med_w_arr.begin(), num_med_w_arr.end());
    std::sort(num_med_h_arr.begin(), num_med_h_arr.end());
    // calculo la mitad
    uint ix_mitad_x = ceil(((float)num_med_cx_arr.size())/2.0f);
    uint ix_mitad_y = ceil(((float)num_med_cy_arr.size())/2.0f);
    uint ix_mitad_w = ceil(((float)num_med_w_arr.size())/2.0f);
    uint ix_mitad_h = ceil(((float)num_med_h_arr.size())/2.0f);
    num_med_cx = num_med_cx_arr[ix_mitad_x];
    num_med_cy = num_med_cy_arr[ix_mitad_y];
    num_med_w = num_med_w_arr[ix_mitad_w];
    num_med_h = num_med_h_arr[ix_mitad_h];
    
    num_avg_w = num_avg_w / ((float)letras_total);
    num_avg_h = num_avg_h / ((float)letras_total);
    std::cout << "characters avgs: " << num_avg_w << " x " << num_avg_h << std::endl;
    std::cout << "characters median center: " << num_med_cx << " x " << num_med_cy << std::endl;
    std::cout << "characters median size: " << num_med_w << " x " << num_med_h << std::endl;

    // 1.2. Buscar las letras conectadas partiendo desde el centro segun la mediana
    std::vector<SegRes> letrasConectadas;

    // funcion que saque del vector original los elementos cercanos
    // Se inicializan las banderas
    for (uint k=0; k<letras_total; k++) {
        SegRes temp = ocrCharaters[k];
        temp.checked = false;
        temp.connected = false;
    }
    
    // Se favorce el estar conectado en el eje X
    float DISTANCIA_CONECTADA_X = num_med_w*2;
    float DISTANCIA_CONECTADA_Y = num_med_h*0.5;
    float UMBRAL_ESPACIO = 0;

    computeConnectedCharacters(ocrCharaters, letrasConectadas, num_med_cx, num_med_cy, DISTANCIA_CONECTADA_X, DISTANCIA_CONECTADA_Y);
    uint conectadosSinCheck;
    do {
        conectadosSinCheck = 0;
        std::vector<uint> letrasActuales;
        for (uint k=0; k<letrasConectadas.size(); k++) {
            SegRes temp = letrasConectadas[k];
            if (temp.connected && !temp.checked) {
                letrasActuales.push_back(k);
                conectadosSinCheck++;
            }
        }
        for (uint k=0; k<letrasActuales.size(); k++) {
            uint indice = letrasActuales[k];
            SegRes temp = letrasConectadas[indice];
            computeConnectedCharacters(ocrCharaters, letrasConectadas, temp.cx, temp.cy, DISTANCIA_CONECTADA_X, DISTANCIA_CONECTADA_Y);
            letrasConectadas[indice].checked = true;
        }
    } while(conectadosSinCheck > 0);

    // Se deben agregar los espacios
    // Se organiza de izquierda a derecha
    std::sort(letrasConectadas.begin(), letrasConectadas.end(), firstSegResLeft);
    // Se evalua la distancia entre el caracter actual y el siguiente
    uint tamanioConectados = letrasConectadas.size();
    std::string completo = "";
    for (uint k=0; k<tamanioConectados; k++) {
        SegRes temp = letrasConectadas[k];
        if (k < tamanioConectados - 1) {
            // Si no es el último caracter
            SegRes siguiente = letrasConectadas[k+1];
            // Calculo la distancia
            int espacio = ((int)siguiente.xi - (int)temp.xf);
            //std::cout << "Espacio entre " << temp.c << siguiente.c << " es " << espacio << std::endl;
            if (espacio > UMBRAL_ESPACIO) {
                completo+=temp.c+" ";
            } else {
                completo += temp.c;
            }
        } else {
            completo += temp.c;
        }
    }

    return completo;
}

std::string ocrFunNum(
    cv::Mat dest, 
    float* ROI_ID,
    std::vector<std::string> class_names, 
    std::string modelPathString,
    std::string modeString,
    int normalize,
    float scoreThreshold,
    float sth,
    float nmsth,
    std::string outfolder
    ) {
    float ROI_ID_X1 = ROI_ID[0];
    float ROI_ID_Y1 = ROI_ID[1];
    float ROI_ID_X2 = ROI_ID[2];
    float ROI_ID_Y2 = ROI_ID[3];
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
        outfolder + "num_");
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

float computeVariance(std::vector<float> samples)
{
     int size = samples.size();

     float variance = 0;
     float t = samples[0];
     for (int i = 1; i < size; i++)
     {
          t += samples[i];
          float diff = ((i + 1) * samples[i]) - t;
          variance += (diff * diff) / ((i + 1.0) *i);
     }

     return variance / (size - 1);
}

// , std::vector& rh, std::vector& gh, std::vector& bh
float computeHistogram(cv::Mat src) {
    std::vector<cv::Mat> bgr_planes;
    cv::split( src, bgr_planes );
    int histSize = 10;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
    cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    std::vector<float> varianzas;
    float sumaVarianza = 0;
    for (uint i = 1; i < histSize; i++ ) {
        float r = r_hist.at<float>(i-1);
        float g = g_hist.at<float>(i-1);
        float b = b_hist.at<float>(i-1);
        std::vector<float> colores;
        colores.push_back(r);
        colores.push_back(g);
        colores.push_back(b);
        float variance = computeVariance(colores);
        //std::cout << variance << ";" << r << ";" << g << ";" << b << std::endl;
        varianzas.push_back(variance);
        sumaVarianza += variance;
    }

    return sumaVarianza / (float) histSize;
    /*
    std::sort(varianzas.begin(), varianzas.end());
    // calculo la mitad
    uint varianzasMitad = ceil(((float)varianzas.size())/2.0f);

    float medianaVarianza = varianzas[varianzasMitad];
    //std::cout << "VarianzaColor:" << medianaVarianza << std::endl;
    return medianaVarianza;
    */
}

double medirDistancia(unsigned char *input, uint step, int i, int j, unsigned char r, unsigned char g, unsigned char b) {
    unsigned char r2 = input[step * j + i ] ;
    unsigned char g2 = input[step * j + i + 1] ;
    unsigned char b2 = input[step * j + i + 2] ;
    double distance = sqrt(pow(r2 - r, 2) + pow(g2 - g, 2) + pow(b2 - b, 2));
    return distance;
}

std::vector<double> computeHistogramOfComponent(cv::Mat src) {

    cv::Mat hsv;
    std::vector<cv::Mat> channels;
    cvtColor(src,hsv,cv::COLOR_BGR2HSV_FULL);
    split(hsv,channels);
    cv::Mat saturation = channels[1];
    cv::Mat hue = channels[0];
    std::vector<unsigned char> hueTodos;

    double ravg = 0;
    double gavg = 0;
    double bavg = 0;
    double count = 0;
    double distanciaPromedio = 0;
    double distanciaPromedioNo = 0;
    double temp;
    unsigned char *input = (unsigned char*)(src.data);
    unsigned char *saturationPixel = (unsigned char*)(saturation.data);
    unsigned char *huePixel = (unsigned char*)(hue.data);
    uint step = src.step;
    for(int j = 0;j < src.rows-1;j++){
        for(int i = 0;i < src.cols-1;i++){
            unsigned char r = input[step * j + i ] ;
            unsigned char g = input[step * j + i + 1] ;
            unsigned char b = input[step * j + i + 2] ;
            float importancia = ((float)saturationPixel[saturation.step * j + i ])/255.0;
            unsigned char theHue = huePixel[hue.step * j + i ];
            hueTodos.push_back(theHue);
            
            temp = medirDistancia(input, step, i+1, j+1, r, g, b);
            distanciaPromedio += importancia*temp;
            distanciaPromedioNo += temp;
            count++;

            temp = medirDistancia(input, step, i, j+1, r, g, b);
            distanciaPromedio += importancia*temp;
            distanciaPromedioNo += temp;
            count++;

            temp = medirDistancia(input, step, i+1, j, r, g, b);
            distanciaPromedio += importancia*temp;
            distanciaPromedioNo += temp;
            count++;

            if (i>0) {
                temp = medirDistancia(input, step, i-1, j+1, r, g, b);
                distanciaPromedio += importancia*temp;
                distanciaPromedioNo += temp;
                count++;
            }
            if (j>0) {
                temp = medirDistancia(input, step, i+1, j-1, r, g, b);
                distanciaPromedio += importancia*temp;
                distanciaPromedioNo += temp;
                count++;
            }
        }
    }

    double sum = 0;
    std::for_each (std::begin(hueTodos), std::end(hueTodos), [&](const unsigned char d) {
        sum += d;
    });
    double mediaHue =  sum / hueTodos.size();

    double accum = 0.0;
    std::for_each (std::begin(hueTodos), std::end(hueTodos), [&](const unsigned char d) {
        accum += (d - mediaHue) * (d - mediaHue);
    });
    double stdev = sqrt(accum / (hueTodos.size()-1));

    distanciaPromedio = distanciaPromedio / count;
    distanciaPromedioNo = distanciaPromedioNo / count;

    std::vector<double> salida;
    salida.push_back(distanciaPromedio);
    salida.push_back(distanciaPromedioNo);
    salida.push_back(stdev);
    return  salida;
}

cv::Mat equalizeBGRA(const cv::Mat& inputImage)
{
    cv::Mat ycrcb;
    cv::Mat result;
    std::vector<cv::Mat> channels;

    result = inputImage;
    
    cvtColor(result,ycrcb,cv::COLOR_BGR2YCrCb);//COLOR_BGR2YCrCb COLOR_BGR2HSV
    split(ycrcb,channels);
    equalizeHist(channels[0], channels[0]);
    equalizeHist(channels[1], channels[1]);
    equalizeHist(channels[2], channels[2]);
    merge(channels,ycrcb);
    cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);//COLOR_YCrCb2BGR COLOR_HSV2BGR
    
    /*
    cvtColor(result,ycrcb,cv::COLOR_BGR2HSV_FULL);//COLOR_BGR2YCrCb COLOR_BGR2HSV
    split(ycrcb,channels);
    equalizeHist(channels[2], channels[2]);
    merge(channels,ycrcb);
    cvtColor(ycrcb, result, cv::COLOR_HSV2BGR_FULL);//COLOR_YCrCb2BGR COLOR_HSV2BGR
    */

    return result;
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
    std::string signaturePathEq;
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
    signaturePathEq = outfolder + "signature_eq.jpg";
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
        ROI_ID,
        class_names, 
        modelPathString,
        modeString,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        outfolder
    );

    std::string myLastName = ocrFunTxt(
        dest, 
        ROI_LAST_NAME,
        class_names4, 
        modelPathString4,
        modeString,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        outfolder + "last_"
    );

    std::string myName = ocrFunTxt(
        dest, 
        ROI_NAME,
        class_names4, 
        modelPathString4,
        modeString,
        normalize,
        scoreThreshold,
        sth,
        nmsth,
        outfolder + "name_"
    );

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
    int signatureWidth = 470;
    int signatureHeight = 158;
    cv::Mat signatureImage = cutImage(dest, sourcePointsSign, signatureWidth, signatureHeight);
    cv::imwrite(signaturePath.c_str(), signatureImage);

    cv::Mat checkImage = dest(cv::Range(dest.size().height*0.5,dest.size().height), cv::Range(0,dest.size().width*0.5));
    cv::Mat equalizedSignature = equalizeBGRA(checkImage);
    cv::Mat checkImageMedian;
    cv::medianBlur(equalizedSignature, checkImageMedian, 15);
    
    cv::Mat equalizedSignatureScaled;
    cv::resize(checkImageMedian, equalizedSignatureScaled, cv::Size(10, 6), 0, 0, cv::INTER_LINEAR);
    cv::imwrite(signaturePathEq.c_str(), equalizedSignatureScaled);
    
    std::vector<double> valores = computeHistogramOfComponent(equalizedSignatureScaled);
    std::cout << "stddev" << (int)valores[2] << std::endl;
    int esFotocopia = 0;
    if (valores[2] < 60) {
        esFotocopia = 1;
    }

    std::ofstream myfile;
    myfile.open(jsonPath.c_str(), std::ios::trunc);

    myfile << "{\n\t\"old_id\":\"";
    myfile << numCedula;
    myfile << "\",\n\t\"id\":\"";
    myfile << myOCR;
    myfile << "\",\n\t\"names\":\"";
    myfile << myName;
    myfile << "\",\n\t\"lastnames\":\"";
    myfile << myLastName;
    myfile << "\",\n\t\"photoPath\":\"";
    myfile << photoPath;
    myfile << "\",\n\t\"signaturePath\":\"";
    myfile << signaturePath;
    myfile << "\",\n\t\"normalized\":\"";
    myfile << normalizedImage;
    myfile << "\",\n\t\"color1\":\"";
    myfile << (int)valores[0];
    myfile << "\",\n\t\"color2\":\"";
    myfile << (int)valores[1];
    myfile << "\",\n\t\"color3\":\"";
    myfile << (int)valores[2];
    myfile << "\",\n\t\"fotocopia\":\"";
    myfile << esFotocopia;
    myfile << "\"\n}" << std::endl;

    myfile.close();
}