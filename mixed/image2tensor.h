
#ifndef __image_2_tensor__
#define __image_2_tensor__

template <typename T>
void genericNormalize(cv::Mat inputImg, uint nChanells, uint type)
{
  T *pixel = inputImg.ptr<T>(0, 0);
  int rows = inputImg.rows;
  int cols = inputImg.cols;
  for (uint i = 0; i < rows; i++)
  {
    for (uint j = 0; j < cols; j++)
    {
      for (uint z = 0; z < nChanells; z++)
      {
        const uint indice = (j * rows + i) * nChanells + z;
        if (type == 10)
        {
          pixel[indice] = (pixel[indice] / 255.0);
        }
        else if (type == 11)
        {
          pixel[indice] = ((pixel[indice] / 255.0) - 0.5) * 2.0;
        }
      }
    }
  }
}

template <typename T>
auto matPreprocess(cv::Mat src, uint width, uint height, uint nChanells, uint type) -> cv::Mat
{
  cv::Mat dst;
  if (type == 10 || type == 11)
  {
    // std::cout << "convertTo CV_32FC3" << std::endl;
    src.convertTo(dst, CV_32FC3);
  }
  else if (type == 256)
  {
    // std::cout << "clone" << std::endl;
    dst = src.clone();
  }
  if (nChanells == 3)
  {
    // Only if RGB swap R with B
    // std::cout << "cvtColor COLOR_BGR2RGB" << std::endl;
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
  }
  if (type == 10 || type == 11)
  {
    // std::cout << "genericNormalize" << std::endl;
    genericNormalize<T>(dst, nChanells, type);
  }
  // std::cout << "resize" << std::endl;
  cv::resize(dst, dst, cv::Size(width, height), 0, 0, cv::INTER_AREA);
  return dst;
}

template <typename T>
void image2tensor(cv::Mat image, TfLiteTensor *input_tensor, uint WIDTH_M, uint HEIGHT_M, uint CHANNEL_M, uint type)
{
  cv::Mat inputImg;
  inputImg = matPreprocess<T>(image, WIDTH_M, HEIGHT_M, CHANNEL_M, type);
  memcpy(input_tensor->data.f, inputImg.ptr<T>(0),
         WIDTH_M * HEIGHT_M * CHANNEL_M * sizeof(T));
  /*
  T *pixel = inputImg.ptr<T>(0);
  for (uint y = 0; y < HEIGHT_M; y++)
  {
    for (uint x = 0; x < WIDTH_M; x++)
    {
      for (uint z = 0; z < CHANNEL_M; z++)
      {
        uint xind = (y * WIDTH_M + x) * CHANNEL_M + z;
        uint xind2 = (x * HEIGHT_M + y) * CHANNEL_M + z;
        input_tensor->data.f[xind] = pixel[xind2];
      }
    }
  }
  */
  return;
}

#endif