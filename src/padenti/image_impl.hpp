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
#include <padenti/image.hpp>


template <typename type, unsigned int nChannels>
Image<type, nChannels>::Image(unsigned int width, unsigned int height):
  m_width(width), m_height(height)
{
  m_data = new type[m_width*m_height*nChannels];
}

template <typename type, unsigned int nChannels>
Image<type, nChannels>::Image(const type *data, unsigned int width, unsigned int height):
  m_width(width), m_height(height)
{
  m_data = new type[m_width*m_height*nChannels];
  std::copy(data, data+m_width*m_height*nChannels, m_data);
}

template <typename type, unsigned int nChannels>
Image<type, nChannels>::Image(const Image<type, nChannels> &image):
  m_width(image.m_width),
  m_height(image.m_height)
{
  m_data = new type[m_width*m_height*nChannels];
  std::copy(image.m_data, image.m_data+image.m_width*image.m_height*nChannels, m_data);
}

template <typename type, unsigned int nChannels>
Image<type, nChannels> &Image<type, nChannels>::operator=(const Image<type, nChannels> &other)
{
  if (!m_data)
  {
    // Not initialized image
    m_data = new type[other.getWidth()*other.getHeight()*nChannels];
    m_width = other.getWidth();
    m_height = other.getHeight();
  }
  else
  {
    if (m_width!=other.getWidth() || m_height!=other.getHeight())
    {
      // Reallocation needed
      delete []m_data;
      m_data = new type[other.getWidth()*other.getHeight()*nChannels];
      m_width = other.getWidth();
      m_height = other.getHeight();
    }
  }
  std::copy(other.getData(), other.getData()+m_width*m_height*nChannels, m_data);

  return *this;
}

template <typename type, unsigned int nChannels>
Image<type, nChannels>::~Image()
{
  delete []m_data;
  m_data = NULL;
}


template <typename type, unsigned int nChannels>
type *Image<type, nChannels>::getData() const
{
  return m_data;
}


template <typename type, unsigned int nChannels>
unsigned int Image<type, nChannels>::getWidth() const
{
  return m_width;
}

template <typename type, unsigned int nChannels>
unsigned int Image<type, nChannels>::getHeight() const
{
  return m_height;
}
