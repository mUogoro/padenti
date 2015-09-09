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

#ifndef __TRAINING_SET_IMAGE_HPP
#define __TRAINING_SET_IMAGE_HPP

#include <padenti/image.hpp>

/*!
 * \brief Base class to store a training set image.
 * The class extends the Image class and stores the following additional information:
 * - a vector of sampled unique pixels indices;
 * - a pointer to a labels image, of the same with/height and number of channels of 
 *   stored image, containing the class index to which the associated pixel
 *   belongs to. Labels start from 1 and the 0 value is reserved to represent unlabeled
 *   pixels;
 * - a vector containing the probability distribution of pixels over classes.
 *
 *  \tparam type TrainingSetImage pixels type.
 *  \tparam nChannels Number of image channels
 */
template <typename type, unsigned int nChannels>
class TrainingSetImage: public Image<type, nChannels>
{
protected:
  unsigned int *m_samples;
  unsigned char *m_labels;
  unsigned int m_nSamples;
  unsigned int m_nClasses;
  float *m_priors;
public:
  /*!
   * Create a new training image of size width X height. The values of data, labels and
   * samples parameters are used to fill, respectively, the pixels values, the pixels labels
   * and the sampled pixels indices.
   *
   * \param data Pointer storing the pixels values. Must be of size width X height X nChannels
   * \param width width of the image
   * \param height height of the image
   * \param labels Pointer storing pixels values. Must be of size width X height. Value must be
   *        in the range [0,nClasses]
   * \param nClasses Number of classes used for image labeling.
   * \param samples Pointer storing unique pixels indices. Indices values must be in the range
   *        [0,width*height*nChannels[.
   * \param nSamples Number of entries in the samples parameter.  
   *
   */
  TrainingSetImage(const type *data, unsigned int width, unsigned int height,
		   const unsigned char *labels, unsigned int nClasses,
		   const unsigned int *samples, unsigned int nSamples);

  /*!
   * Copy constructor. Create a new TrainingSetImage with the same with and size of tsImg and
   * copy its pixels values, labels and samples. Pixels, labels and samples internal pointers
   * will be reallocate if the input image size or number of samples are different.
   *
   * \param tsImg Input image whose data will be copied to the new instance.
   *
   */
  TrainingSetImage(const TrainingSetImage<type, nChannels> &tsImg);
  ~TrainingSetImage();

  /*!
   * Get the internal pointer where pixels labels are stored.
   *
   * \return The internal pointer to pixels labels.
   *
   */
  const unsigned char *getLabels() const;

  /*!
   * Get the internal pointer where sampled pixels indices are stored.
   *
   * \return The internal pointer to samples indices.
   *
   */
  const unsigned int *getSamples() const;

  /*!
   * Get the number of sampled indices.
   *
   * \return The number of sampled indices
   *
   */
  unsigned int getNSamples() const;

  /*!
   * Get the probability distribution over classes for sampled pixels.
   *
   * \return A vector where the i-th entry stores the probability of the class i+1.
   *
   */
  const float *getPriors() const;
};


#include <padenti/training_set_image_impl.hpp>

#endif // __TRAINING_SET_IMAGE_HPP
