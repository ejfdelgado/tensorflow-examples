#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

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