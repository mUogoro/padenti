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

#ifndef __IMAGE_LOADER_HPP
#define __IMAGE_LOADER_HPP

#include <string>
#include <padenti/image.hpp>

/*!
 * \brief Class interface for image data loading.
 * The ImageLoader class defines an interface for loading image data from a file.
 *
 *  \tparam type images pixels type.
 *  \tparam nChannels Number of channels for loaded images
 *
 */
template <typename type, unsigned int nChannels>
class ImageLoader
{
public:
  /*!
   * Load the image data stored on disk. The data pointer is filled with the pixel data,
   * while width and height are set to the image channels size.
   *
   * \param imagePath Path of the image to load
   * \param width Image channels width
   * \param height Image channels height
   *
   */
  virtual void load(const std::string &imagePath, type *data,
		    unsigned int *width, unsigned int *height)=0;

  /*!
   * Load the image data stored on disk and return a new Image object.
   *
   * \param imagePath Path of the image to load
   * 
   * \return A new Image instance
   *
   */
  virtual Image<type, nChannels> load(const std::string &imagePath)=0;
};

#endif // __IMAGE_LOADER_HPP
