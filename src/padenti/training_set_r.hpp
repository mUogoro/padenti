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

#ifndef __TRAINING_SET_R_HPP
#define __TRAINING_SET_R_HPP

#include <string>
#include <vector>
#include <padenti/training_set_rimage.hpp>
#include <padenti/image_loader.hpp>
#include <padenti/image_sampler.hpp>

/*!
 * \brief TrainingSet class.
 * The class is responsible of loading and sampling a set of images pairs stored on disk in
 * a path specified by the user. Each pair consists of a data image, containing pixels values,
 * and a labels image, where pixels labels are stored. The two images are combined in a single
 * TrainingSetImage instance, sampled and stored within the TrainingSet object. Pairs are
 * recognized by their names: data and labels image files have the same prefix and a different
 * suffix specified by the user.
 *
 */
template <typename type, unsigned int nChannels, unsigned int rSize>
class TrainingSetR
{
private:
  unsigned int m_nImages;
  std::vector<TrainingSetRImage<type, nChannels, rSize> > m_images;
public:

  /*!
   * Create an empty training set.
   *
   * \param Maximum number of different classes used for labeling
   */
  TrainingSetR();

  /*!
   * Create a new training set. The following steps are performed:
   * - all pairs of files with the same prefix and the suffixes specified by the dataSuffix/
   *   labelsSuffix parameters and stored within tsPath are selected;
   * - dataLoader and labelsLoader ImageLoader instances are used to load, respectively,
   *   the pixels and labels of each pair;
   * - pixels and labels data is sampled using the ImageSampler sampler instance;
   * - pixels, labels and sampled indices are used to create a new TrainingSetImage instance.
   *
   * \param tsPath The training set path where pairs are stored
   * \param dataSuffix File suffix used to identify image files
   * \param labelsSuffix File suffix used to identify labels files
   * \param nClasses Maximum number of different classes used for labeling
   * \param dataLoader ImageLoader instance used to load pixels data
   * \param labelsLoader LabelsLoader instance used to load labels data
   * \param sampler ImageSampler instance used for sampling
   */
  TrainingSetR(const std::vector<std::string> &dataPaths, ImageLoader<type, nChannels> &dataLoader,
	       const std::vector<std::vector<float> > &values, const ImageSampler<type, nChannels> &sampler);
  ~TrainingSetR();

  /*!
   * Get the vector of training set images
   *
   * \return vector of TrainingSetImage instances
   */
  const std::vector<TrainingSetRImage<type, nChannels, rSize> > &getImages() const;
  

  /*!
   * Add a training set image to the training set
   * 
   * \param image a TrainingSetImage instance
   */
  TrainingSetR<type, nChannels, rSize>&
      operator<<(const TrainingSetRImage<type, nChannels, rSize> &image);


  /*!
   * Get the number of images stored in the training set
   *
   * \return the number of training set images
   */
  unsigned int getNImages() const;

  /*!
   * Get the probability distribution over classes for the sampled pixels
   *
   * \return A vector where the i-th entry stores the probability of the class i+1.
   */
  // const float *getPriors() const;
};


#include <padenti/training_set_r_impl.hpp>

#endif // __TRAINING_SET_R_HPP
