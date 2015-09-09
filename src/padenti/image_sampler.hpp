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

#ifndef __IMAGE_SAMPLER_HPP
#define __IMAGE_SAMPLER_HPP

/*!
 *  \brief Class interface for image sampling.
 *  The ImageSampler class defines an interface for image sampling.  
 *
 *  \tparam type Image pixels type.
 *  \tparam nChannels Number of image channels
 */
template <typename type, unsigned int nChannels>
class ImageSampler
{
protected:
  unsigned int m_nSamples;
  unsigned int m_seed;
public: 
  /*!
   * Create a new sampler instance. The number of pixels to sample and an optional
   * seed for random sampling are specified.
   *
   * \param nSamples Number of pixels sampled for each image
   * \param seed Optional seed used for random sampling
   */
  ImageSampler(unsigned int nSamples, unsigned int seed=0):
    m_nSamples(nSamples),
    m_seed(seed){};

  /*!
   * Sample image data. Pixels data is stored continuously in data pointer, whereas labels stores
   * the corresponding pixels labels. Both image channels (i.e. data parameter) and labels images
   * must be of size width X height. Sampled pixels indices are stored within the vector pointed
   * by samples parameter. The number of sampled pixels is returned.
   *
   * \param data image pixels values
   * \param labels image pixels labels
   * \param width image width
   * \param height image height
   * \param samples output vector where sampled pixels indices are stored. Must be at least of
   *        size nSamples (specified by constructor call)
   * \return The number of sampled indices
   *
   */
  virtual unsigned int sample(const type *data, const unsigned char *labels,
			      unsigned int width, unsigned int height, unsigned int *samples) const =0;
};

#endif // __IMAGE_SAMPLER_HPP
