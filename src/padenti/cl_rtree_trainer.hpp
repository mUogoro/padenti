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

#ifndef __CL_RTREE_TRAINER_HPP
#define __CL_RTREE_TRAINER_HPP

#include <string>
#include <boost/unordered_map.hpp>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <padenti/tree_trainer.hpp>


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int RDim>
class CLRTreeTrainer
{
private:
  //cl::Platform m_clPlatform;
  cl::Context m_clContext;
  cl::Device m_clDevice;
  cl::CommandQueue m_clQueue1, m_clQueue2;

  cl::Program m_clHistUpdateProg;
  cl::Program m_clPredictProg;
  cl::Program m_clLearnBestFeatProg;
  cl::Kernel m_clPerImgHistKern;
  cl::Kernel m_clPredictKern;
  cl::Kernel m_clLearnBestFeatKern;
  
  cl::Buffer m_clTreeLeftChildBuff;
  cl::Buffer m_clTreeFeaturesBuff;
  cl::Buffer m_clTreeThrsBuff;
  //cl::Buffer m_clTreePosteriorsBuff;
  cl::Buffer m_clTreeValuesBuff;

  cl::Image   *m_clTsImg1,         *m_clTsImg2;
  cl::Image2D m_clTsMaskImg1,    m_clTsMaskImg2;
  cl::Image2D m_clTsNodesIDImg1,   m_clTsNodesIDImg2;
  cl::Buffer  m_clTsSamplesBuff1,  m_clTsSamplesBuff2;
  cl::Image2D m_clPredictImg1,     m_clPredictImg2;
  cl::Buffer  m_clPerImgHistBuff1, m_clPerImgHistBuff2;
  cl::Buffer m_clTsImgPinn;
  cl::Buffer m_clTsMaskImgPinn;
  cl::Buffer m_clTsNodesIDImgPinn;
  cl::Buffer m_clTsSamplesBuffPinn;
  cl::Buffer m_clPerImgHistBuffPinn;
  ImgType       *m_clTsImgPinnPtr;
  unsigned char *m_clTsMaskImgPinnPtr;
  int           *m_clTsNodesIDImgPinnPtr;
  unsigned int  *m_clTsSamplesBuffPinnPtr;
  unsigned char  *m_clPerImgHistBuffPinnPtr;

  /** \todo move to training set class??? */
  unsigned int m_maxTsImgWidth;
  unsigned int m_maxTsImgHeight;
  unsigned int m_maxTsImgSamples;
  bool *m_skippedTsImg;
  bool *m_toSkipTsImg;

  cl::Buffer m_clFeatLowBoundsBuff;
  cl::Buffer m_clFeatUpBoundsBuff;
  unsigned int *m_perNodeTotSamples;
  int *m_frontier;
  boost::unordered_map<int, int> m_frontierIdxMap;

  cl::Buffer m_clHistogramBuff;
  size_t m_histogramSize;
  double **m_histogram;

  //cl::Buffer m_clBestFeaturesBuff;
  //cl::Buffer m_clBestThresholdsBuff;
  //cl::Buffer m_clBestEntropiesBuff;
  //cl::Buffer m_clPerClassTotSamplesBuff;
  //unsigned int *m_bestFeatures;
  //unsigned int *m_bestThresholds;
  float *m_bestEntropies;

  unsigned int m_seed;

private:
  void _initTrain(RTree<FeatType, FeatDim, RDim> &tree,
		  const TrainingSetR<ImgType, nChannels, RDim> &trainingSet,
		  const TreeTrainerParameters<FeatType, FeatDim> &params,
		  unsigned int startDepth, unsigned int endDepth);
  void _initPerNodeCounters(RTree<FeatType, FeatDim, RDim> &tree,
			    const TrainingSetR<ImgType, nChannels, RDim> &trainingSet,
			    const TreeTrainerParameters<FeatType, FeatDim> &params,
			    unsigned int startDepth, unsigned int endDepth);
  unsigned int _initFrontier(RTree<FeatType, FeatDim, RDim> &tree,
			     const TreeTrainerParameters<FeatType, FeatDim> &params, unsigned int currDepth);
  unsigned int _initHistogram(const TreeTrainerParameters<FeatType, FeatDim> &params);
  void _traverseTrainingSet(const TrainingSetR<ImgType, nChannels, RDim> &trainingSet,
			    const TreeTrainerParameters<FeatType, FeatDim> &params,
			    unsigned int currDepth, unsigned int currSlice);
  void _learnBestFeatThr(RTree<FeatType, FeatDim, RDim> &tree,
			 const TreeTrainerParameters<FeatType, FeatDim> &params,
			 unsigned int currDepth, unsigned int currSlice);
  void _cleanTrain();

public:
  CLRTreeTrainer(const std::string &featureKernelPath, bool useCPU);
  ~CLRTreeTrainer();
  void train(RTree<FeatType, FeatDim, RDim> &tree,
	     const TrainingSetR<ImgType, nChannels, RDim> &trainingSet,
	     const TreeTrainerParameters<FeatType, FeatDim> &params,
	     unsigned int startDepth, unsigned int endDepth);
};


#include <padenti/cl_rtree_trainer_impl.hpp>

#endif // __CL_RTREE_TRAINER_HPP
