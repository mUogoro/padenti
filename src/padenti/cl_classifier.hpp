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

#ifndef __CL_CLASSIFIER_HPP
#define __CL_CLASSIFIER_HPP

#include <string>
#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <padenti/tree.hpp>
#include <padenti/classifier.hpp>


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
class CLClassifier: public Classifier<ImgType, nChannels, FeatType, FeatDim, nClasses>
{
private:
  cl::Context m_clContext;
  cl::Device m_clDevice;
  cl::CommandQueue m_clQueue;

  cl::Program m_clPredictProg;
  cl::Kernel m_clPredictKern;
  cl::Kernel m_clComputePosteriorKern;

  /*
  cl::Buffer m_clTreeLeftChildBuff;
  cl::Buffer m_clTreeFeaturesBuff;
  cl::Buffer m_clTreeThrsBuff;
  cl::Buffer m_clTreePosteriorsBuff;
  */
  std::vector<cl::Buffer> m_clTreeLeftChildBuff;
  std::vector<cl::Buffer> m_clTreeFeaturesBuff;
  std::vector<cl::Buffer> m_clTreeThrsBuff;
  std::vector<cl::Buffer> m_clTreePosteriorsBuff;
  std::vector<unsigned int> m_treeDepth;
  unsigned int m_nTrees;

  cl::Image *m_clImg;
  cl::Image2D m_clMask;
  cl::Image2D m_clNodesIDImg;
  cl::Image2D m_clPredictImg;
  cl::Buffer m_clPosteriorBuff;

  cl::Buffer m_clImgPinn;
  cl::Buffer m_clMaskPinn;
  cl::Buffer m_clPredictImgPinn;
  cl::Buffer m_clPosteriorPinn;

  ImgType *m_clImgPinnPtr;
  unsigned char *m_clMaskPinnPtr;
  int *m_clPredictImgPinnPtr;
  float *m_clPosteriorPinnPtr;
  
  size_t m_internalImgWidth;
  size_t m_internalImgHeight;

  void _initImgObjects(size_t, size_t, bool);

public:
  //CLClassifier(const Tree<FeatType, FeatDim, nClasses> &tree,
  //	       const std::string &featureKernelPath, bool useCPU);
  CLClassifier(const std::string &featureKernelPath, bool useCPU=false);
  ~CLClassifier();

  CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>&
    operator<<(const Tree<FeatType, FeatDim, nClasses>&);

  void predict(unsigned int,
	       const Image<ImgType, nChannels> &image, Image<int, 1> &prediction);
  void predict(unsigned int,
	       const Image<ImgType, nChannels> &image, Image<int, 1> &prediction,
	       Image<unsigned char, 1> &mask);
  void predict(const Image<ImgType, nChannels> &image, Image<float, nClasses> &prediction);
  void predict(const Image<ImgType, nChannels> &image, Image<float, nClasses> &prediction,
	       Image<unsigned char, 1> &mask);
};


#include <padenti/cl_classifier_impl.hpp>

#endif // __CL_CLASSIFIER_HPP
