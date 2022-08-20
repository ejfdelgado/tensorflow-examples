#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <sstream>

#ifndef __utils_h__
#define __utils_h__

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

std::vector<std::string> readLabelsFile(const char *path)
{
  std::vector<std::string> myvector;
  std::fstream newfile;
  newfile.open(path, std::ios::in);
  if (newfile.is_open())
  {
    std::string tp;
    while (getline(newfile, tp))
    {
      myvector.push_back(tp);
    }
    newfile.close();
  }

  return myvector;
}

cv::ImreadModes string2ImreadModesEnum(std::string str)
{
  static std::unordered_map<std::string, cv::ImreadModes> const table = {
      {"IMREAD_GRAYSCALE", cv::ImreadModes::IMREAD_GRAYSCALE},
      {"IMREAD_COLOR", cv::ImreadModes::IMREAD_COLOR}};
  auto it = table.find(str);
  if (it != table.end())
  {
    return it->second;
  }
  else
  {
    return cv::ImreadModes::IMREAD_COLOR;
  }
}

cv::Mat cutImage(cv::Mat &src, std::vector<cv::Point2f> source, uint destWidth, uint destHeight)
{
  cv::Point2f srcTri[4];
  srcTri[0] = source[0];
  srcTri[1] = source[1];
  srcTri[2] = source[2];
  srcTri[3] = source[3];
  
  cv::Point2f dstTri[4];
  dstTri[0] = cv::Point2f(0.f, 0.f);
  dstTri[1] = cv::Point2f(destWidth, 0.f);
  dstTri[2] = cv::Point2f(0, destHeight);
  dstTri[3] = cv::Point2f(destWidth, destHeight);
  cv::Mat warp_dst = cv::Mat::zeros(destHeight, destWidth, src.type());

  cv::Mat warp_mat = cv::getPerspectiveTransform(srcTri, dstTri);
  cv::warpPerspective(src, warp_dst, warp_mat, warp_dst.size());

  return warp_dst;
}

#endif