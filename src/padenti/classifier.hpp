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

#ifndef __CLASSIFIER_HPP
#define __CLASSIFIER_HPP

#include <padenti/tree.hpp>
#include <padenti/image.hpp>

/*!
 * \brief Class interface for classification
 * The Classifier class defines an interface for concrete classes implementing the
 * Random Forests classifier.
 * 
 * \tparam type Image pixels type.
 * \tparam nChannels Number of image channels
 * \tparam FeatType type of feature entries and threshold
 * \tparam FeatDim dimension (i.e. number of entries) of the feature
 * \tparam nClasses number of classes
 *
 * \todo define tree loading methods here as well
 */
template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
class Classifier
{
public:
  /*!
   * Perform prediction using the tree with id ID on the image image. The prediction image
   * will store the indices of the leaf node reached at each pixel.
   *
   * \param id tree unique id
   * \param image input image
   * \param prediction integer image which store prediction results as leaf node indices
   */
  virtual void predict(unsigned int id,
		       const Image<ImgType, nChannels> &image, Image<int, 1> &prediction)=0;

  /*!
   * Perform prediction using the tree with id ID on the image image. The prediction image
   * will store the indices of the leaf node reached at each pixel. The prediction is
   * performed only on the pixels whose corresponding mask value is different from zero.
   *
   * \param id tree unique id
   * \param image input image
   * \param prediction integer image which stores prediction results as leaf node indices
   * \param mask binary image where a non-zero value means that the corresponding pixel on
   *        image must be processed
   */
  virtual void predict(unsigned int id,
		       const Image<ImgType, nChannels> &image, Image<int, 1> &prediction,
		       Image<unsigned char, 1> &mask)=0;
  
  /*!
   * Perform prediction using the whole trees ensemble on the image image. prediction is a 
   * multichannel float image, one channel per class, where each pixel stores the posterior
   * probability for the correspondent class.
   *
   * \param image input image
   * \param prediction float image which stores prediction results as per-class posterior
   *        probability
   */
  virtual void predict(const Image<ImgType, nChannels> &image, Image<float, nClasses> &prediction)=0;

  /*!
   * Perform prediction using the whole trees ensemble on the image image. prediction is a 
   * multichannel float image, one channel per class, where each pixel stores the posterior
   * probability for the correspondent class. The prediction is performed only on the
   * pixels whose corresponding mask value is different from zero.
   *
   * \param image input image
   * \param prediction float image which stores prediction results as per-class posterior
   *        probability
   * \param mask binary image where a non-zero value means that the corresponding pixel on
   *        image must be processed
   */
  virtual void predict(const Image<ImgType, nChannels> &image, Image<float, nClasses> &prediction,
		       Image<unsigned char, 1> &mask)=0;
  virtual ~Classifier(){};
};


#endif // __CLASSIFIER_HPP
