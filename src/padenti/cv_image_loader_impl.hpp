/******************************************************************************
 * Padenti Library
 *
 * Copyright (C) 2015  Daniele Pianu <daniele.pianu@ieiit.cnr.it>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 ******************************************************************************/

#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <padenti/cv_image_loader.hpp>


template <typename type, unsigned int nChannels>
inline bool _checkCVImgType(const cv::Mat img)
{
  return cv::DataType<type>::depth==img.depth() && img.channels()==nChannels;
}



template <typename type, unsigned int nChannels>
void CVImageLoader<type, nChannels>::load(const std::string &imagePath, type *data,
					  unsigned int *width, unsigned int *height)
{
  cv::Mat img = cv::imread(imagePath, -1);

  if (_checkCVImgType<type, nChannels>(img))
  {
    *width = img.cols;
    *height = img.rows;
    std::copy((type*)img.data, ((type*)img.data)+(*width)*(*height)*nChannels, data);
  }
  else
  {
    throw "Image type different from expected one";
  }
}

template <typename type, unsigned int nChannels>
Image<type, nChannels> CVImageLoader<type, nChannels>::load(const std::string &imagePath)
{
  cv::Mat img = cv::imread(imagePath, -1);

  if (_checkCVImgType<type, nChannels>(img))
  {
    Image<type, nChannels> retImg((const type*)img.data, img.cols, img.rows);
    return retImg;
  }
  else
  {
    throw "Image type different from expected one";
  }
}



CVRGBLabelsLoader::CVRGBLabelsLoader(const unsigned char (*rgb2labelMap)[3], unsigned int nClasses):
m_nClasses(nClasses)
{
  m_rgb2labelMap = new unsigned char[nClasses][3];
  for (unsigned int i=0; i<m_nClasses; i++)
  {
    std::copy(rgb2labelMap[i], rgb2labelMap[i]+3, m_rgb2labelMap[i]);
  }
}


void _rgb2label(const cv::Mat &src, unsigned char *dst, const unsigned char (*rgb2labelMap)[3],
		unsigned int nClasses);
void CVRGBLabelsLoader::load(const std::string &imagePath, unsigned char *data,
			     unsigned int *width, unsigned int *height)
{
  cv::Mat img = cv::imread(imagePath, -1);
  if (img.type()==CV_8UC3 || img.type()==CV_8UC4)
  {
    cv::Mat rgbImg;

    *width = img.cols;
    *height = img.rows;
    cv::cvtColor(img, rgbImg, img.channels()==3 ? CV_BGR2RGB : CV_BGRA2RGBA);
    _rgb2label(rgbImg, data, m_rgb2labelMap, m_nClasses);
  }
  else
  {
    throw "Labels image type must be CV_8UC3 or CV_8UC4";
  }
}


Image<unsigned char, 1> CVRGBLabelsLoader::load(const std::string &imagePath)
{
  cv::Mat img = cv::imread(imagePath, -1);
  if (img.type()==CV_8UC3 || img.type()==CV_8UC4)
  {
    cv::Mat rgbImg;
    Image<unsigned char, 1> retImg(img.cols, img.rows);
    
    cv::cvtColor(img, rgbImg, img.channels()==3 ? CV_BGR2RGB : CV_BGRA2RGBA);
    _rgb2label(rgbImg, retImg.getData(), m_rgb2labelMap, m_nClasses);
    
    return retImg;
  }
  else
  {
    throw "Labels image type must be CV_8UC3 or CV_8UC4";
  }
}



CVRGBLabelsLoader::~CVRGBLabelsLoader()
{
  //for (int i=0; i<m_nClasses; i++) delete[]m_rgb2labelMap[i];
  delete []m_rgb2labelMap;
}



// Note: code for countours retrieval from
// http://docs.opencv.org/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html
/**
 * \todo adapt code to specific case
 */
cv::Rect _extractROI(const cv::Mat &img);

template <typename type, unsigned int nChannels>
void CVImageROILoader<type, nChannels>::load(const std::string &imagePath, type *data,
					     unsigned int *width, unsigned int *height)
{
  cv::Mat img = cv::imread(imagePath, -1);

  if (_checkCVImgType<type, nChannels>(img))
  {
    cv::Rect boundRect = _extractROI(img);
    cv::Mat tmp(boundRect.height, boundRect.width, CV_MAKETYPE(cv::DataType<type>::depth, nChannels),
		(void*)data);

    *width = boundRect.width;
    *height = boundRect.height;
    m_roiX=boundRect.x;
    m_roiY=boundRect.y;
    img(boundRect).copyTo(tmp);
  }
  else
  {
    throw "Image type different from expected one";
  }
}


template <typename type, unsigned int nChannels>
Image<type, nChannels> CVImageROILoader<type, nChannels>::load(const std::string &imagePath)
{
  cv::Mat img = cv::imread(imagePath, -1);

  if (_checkCVImgType<type, nChannels>(img))
  {
    cv::Rect boundRect = _extractROI(img);
    Image<type, nChannels> retImg(boundRect.width, boundRect.height);

    cv::Mat tmp(boundRect.height, boundRect.width, CV_MAKETYPE(cv::DataType<type>::depth, nChannels),
		(void*)retImg.getData());
    img(boundRect).copyTo(tmp);
    m_roiX=boundRect.x;
    m_roiY=boundRect.y;
    return retImg;
  }
  else
  {
    throw "Image type different from expected one";
  }
  
}



CVRGBLabelsROILoader::CVRGBLabelsROILoader(const unsigned char (*rgb2labelMap)[3], unsigned int nClasses):
  CVRGBLabelsLoader(rgb2labelMap, nClasses){}


/** \todo rewrite to take advantage of _extractROI helper function */
void CVRGBLabelsROILoader::load(const std::string &imagePath, unsigned char *data,
				unsigned int *width, unsigned int *height
)
{
  CVRGBLabelsLoader::load(imagePath, data, width, height);

  cv::Mat tmp(*height, *width, CV_8UC1, (void*)data);
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Rect boundRect;

  cv::Mat mask = tmp>0;
  cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  boundRect = cv::boundingRect(contours[0]);

  cv::Mat out(boundRect.height, boundRect.width, CV_8UC1, (void*)data);
  cv::Mat roi = tmp(boundRect).clone();

  *width = boundRect.width;
  *height = boundRect.height;
  m_roiX=boundRect.x;
  m_roiY=boundRect.y;

  roi.copyTo(out);
}


/** \todo Optimize to avoid temporary Image object creation */
Image<unsigned char, 1> CVRGBLabelsROILoader::load(const std::string &imagePath)
{
  Image<unsigned char, 1> img(CVRGBLabelsLoader::load(imagePath));
  cv::Mat tmp = cv::Mat(img.getHeight(), img.getWidth(),
			CV_MAKETYPE(cv::DataType<unsigned char>::depth, 1), (void*)img.getData());
  
  cv::Rect boundRect = _extractROI(tmp);
  
  Image<unsigned char, 1> retImg ((unsigned char*)tmp(boundRect).clone().data, boundRect.width, boundRect.height);
  m_roiX=boundRect.x;
  m_roiY=boundRect.y;


  return retImg;
}



void _rgb2label(const cv::Mat &src, unsigned char *dst, const unsigned char (*rgb2labelMap)[3],
		unsigned int nClasses)
{
  for (unsigned int v=0; v<src.rows; v++)
  {
    for (unsigned int u=0; u<src.cols; u++)
    {
      unsigned char l=0;
      if (src.channels()==3)
      {
	cv::Vec3b labelValue = src.at<cv::Vec3b>(v, u);
	for (; l<nClasses; l++)
	{
	  if (labelValue[0]==rgb2labelMap[l][0] &&
	      labelValue[1]==rgb2labelMap[l][1] &&
	      labelValue[2]==rgb2labelMap[l][2]) break;
	}
	dst[v*src.cols+u] = (l<nClasses) ? l+1 : 0;
      }
      else
      {
	cv::Vec4b labelValue = src.at<cv::Vec4b>(v, u);
	for (; l<nClasses; l++)
	{
	  if (labelValue[0]==rgb2labelMap[l][0] &&
	      labelValue[1]==rgb2labelMap[l][1] &&
	      labelValue[2]==rgb2labelMap[l][2]) break;
	}
	dst[v*src.cols+u] = (l<nClasses) ? l+1 : 0;
      }
    }
  }
}



cv::Rect _extractROI(const cv::Mat &img)
{
   std::vector<std::vector<cv::Point> > contours;
   std::vector<cv::Vec4i> hierarchy;
   cv::Rect boundRect;

   cv::Mat mask = img>0;
   cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
   boundRect = cv::boundingRect(contours[0]);
   
   return boundRect;
}
