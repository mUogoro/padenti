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

#ifndef __CV_IMAGE_LOADER_HPP
#define __CV_IMAGE_LOADER_HPP


#include <padenti/image_loader.hpp>


template <typename type, unsigned int nChannels>
class CVImageLoader: public ImageLoader<type, nChannels>
{
public:
  void load(const std::string &imagePath, type *data,
	    unsigned int *width, unsigned int *height);
  Image<type, nChannels> load(const std::string &imagePath);
};


class CVRGBLabelsLoader: public ImageLoader<unsigned char, 1>
{
private:
  unsigned char (*m_rgb2labelMap)[3];
  unsigned int m_nClasses;
public:
  CVRGBLabelsLoader(const unsigned char (*rgb2labelMap)[3], unsigned int nClasses);
  ~CVRGBLabelsLoader();
  void load(const std::string &imagePath, unsigned char *data,
	    unsigned int *width, unsigned int *height);
  Image<unsigned char, 1> load(const std::string &imagePath);
};


/** \todo template for generalized cropping */

template <typename type, unsigned int nChannels>
class CVImageROILoader: public CVImageLoader<type, nChannels>
{
private:
  unsigned int m_roiX;
  unsigned int m_roiY;
public:
  void load(const std::string &imagePath, type *data,
	    unsigned int *width, unsigned int *height);
  Image<type, nChannels> load(const std::string &imagePath);

  unsigned int getRoiX() const {return m_roiX;}
  unsigned int getRoiY() const {return m_roiY;}
};


class CVRGBLabelsROILoader: public CVRGBLabelsLoader
{
private:
  unsigned int m_roiX;
  unsigned int m_roiY;
public:
  CVRGBLabelsROILoader(const unsigned char (*rgb2labelMap)[3], unsigned int nClasses);
  void load(const std::string &imagePath, unsigned char *data,
	    unsigned int *width, unsigned int *height);
  Image<unsigned char, 1> load(const std::string &imagePath);

  unsigned int getRoiX() const {return m_roiX;}
  unsigned int getRoiY() const {return m_roiY;}
};


#include <padenti/cv_image_loader_impl.hpp>

#endif // __CV_IMAGE_LOADER_HPP
