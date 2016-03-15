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

#ifndef __TREE_TRAINER_HPP
#define __TREE_TRAINER_HPP

#include <padenti/tree.hpp>
#include <padenti/image.hpp>
#include <padenti/image_sampler.hpp>
#include <padenti/training_set.hpp>


/*!
 * \brief Class representing training parameters
 *
 * \tparam FeatType type of feature entries and threshold
 * \tparam FeatDim dimension (i.e. number of entries) of the feature
 */
template <typename FeatType, unsigned int FeatDim>
class TreeTrainerParameters
{
public:
  unsigned int nFeatures;          /*!< Number of per-pixel sampled features */
  unsigned int nThresholds;        /*!< Number of per-feature sampled thresholds */
  bool computeFRange;              /*!< Not used */
  FeatType featLowBounds[FeatDim]; /*!< Lower bound (i.e. minimum valid value)
				     for each feature entry */
  FeatType featUpBounds[FeatDim];  /*!< Upper bound (i.e. maximum valid value)
				     for each feature entry */
  FeatType thrLowBound;            /*!< Minimum valid threshold value */
  FeatType thrUpBound;             /*!< Maximum valid threshold value */
  bool randomThrSampling;          /*!< Number of pixels sampled from each image */
  unsigned int perLeafSamplesThr;  /*!< Mininum number of pixels at each leaf node
				     to allow further splitting */
  unsigned int nFeatLutSamples;   /*!< Number of entries to sample from the features LUT*/
  std::vector<FeatType> featLut;   /*!< LookUp Table with features values, stored
				     in row-major format */
  std::vector<FeatType> thrLut;    /*!< LookUp Table with threshold values, stored
				     in row-major format */
};


/*!
 * \brief Class interface for tree training
 * The TreeTrainer class defines an interface for concrete classes implementing the training
 * of a single tree of the Random Forests. Concrete classes must implement the train method
 * that, given an imput TrainingSet instance, a set of parameters specified by a
 * TreeTrainerParameters instance and a maximum depth, is responsible of tree training.
 *
 * \tparam type Image pixels type.
 * \tparam nChannels Number of image channels
 * \tparam FeatType type of feature entries and threshold
 * \tparam FeatDim dimension (i.e. number of entries) of the feature
 * \tparam nClasses number of classes
 */
template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
class TreeTrainer
{
public:
  /*!
   * Train a single tree of the Random Forests ensemble up to depth endDepth on the trainintSet
   * training set
   * 
   * \param tree the tree to be trained
   * \param trainingSet the input training set
   * \param params the training parameters
   * \param startDepth must be 1
   * \param endDepth the maximum depth at which training stops
   */
  virtual void train(Tree<FeatType, FeatDim, nClasses> &tree,
		     const TrainingSet<ImgType, nChannels> &trainingSet,
		     const TreeTrainerParameters<FeatType, FeatDim> &params,
		     unsigned int startDepth, unsigned int endDepth)=0;
};

#endif // __TREE_TRAINER_HPP
