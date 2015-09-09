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

#ifndef __IMAGE_HPP
#define __IMAGE_HPP

#include <cstdlib>

/*! 
 *  \brief Base class to store an image.
 *  The class acts as a simple container of pixels values. Arbitrary pixels type and number
 *  of channels are supported through template parameters. Image pixels must be stored
 *  continuously in memory.
 *
 *  \tparam type Image pixels type.
 *  \tparam nChannels Number of image channels
 */
template <typename type, unsigned int nChannels>
class Image
{
protected:
  type *m_data;
  unsigned int m_width;
  unsigned int m_height;
public:
  /*!
   * Base constructor. Internal pointer is initialized to NULL. To be used, the instance
   * must be initialized using the copy constructor.
   */
  Image():m_data(NULL){};

  /*!
   * Create a new image of size width X height and fill it using the pixels values in
   * data parameter.
   *
   * \param data Pointer storing the pixels values. Must be of size width X height X nChannels
   * \param width width of the image
   * \param height height of the image
   *
   */
  Image(const type *data, unsigned int width, unsigned int height);

  /*!
   * Create a new image of size width X height. Pixels values are left unitialized.
   *
   * \param width width of the image
   * \param height height of the image
   *
   */
  Image(unsigned int width, unsigned int height);

  /*!
   * Copy constructor. Create a new image with the same size of image parameters and
   * copy its pixel values.
   *
   * \param image Input image whose pixels will be copied to the new instance.
   *
   */
  Image(const Image<type, nChannels> &image);
  ~Image();

  /*!
   * Copy the pixels values of other into the current image. If the current image size is
   * different, free its internal pointer and allocate a new one with the same with and
   * height of other.
   *
   * \param other Another Image instance
   *
   */
  Image<type, nChannels>& operator=(const Image<type, nChannels> &other);

  /*!
   * Get the internal pointer where pixels values are stored.
   *
   * \return The internal pointer to pixels values.
   *
   */
  type *getData() const;
  
  /*!
   * Get image width.
   *
   * \return Image width
   *
   */
  unsigned int getWidth() const;

  /*!
   * Get image height.
   *
   * \return Image height
   *
   */
  unsigned int getHeight() const;
};

#include <padenti/image_impl.hpp>

#endif // __IMAGE_HPP
