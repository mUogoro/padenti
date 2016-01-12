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

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <padenti/cl_feat_fmt_traits.hpp>
#include <padenti/cl_img_fmt_traits.hpp>
#include <padenti/classifier.hpp>

#include <padenti/predict.cl.inc>

#define WG_WIDTH (16)
#define WG_HEIGHT (16)


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::CLClassifier(
  //const Tree<FeatType, FeatDim, nClasses> &tree,
  const std::string &featureKernelPath,
  bool useCPU):
  //m_depth(tree.getDepth())
  m_nTrees(0)
{
  m_clContext = cl::Context(useCPU ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU);
  m_clDevice = m_clContext.getInfo<CL_CONTEXT_DEVICES>()[0];
  m_clQueue = cl::CommandQueue(m_clContext, m_clDevice, 0);

  std::string clPredictStr(reinterpret_cast<const char*>(const_cast<const unsigned char*>(predict_cl)),
			   predict_cl_len);
  cl::Program::Sources clPredictSrc(1, std::make_pair(clPredictStr.c_str(),
						      clPredictStr.length()+1));

  m_clPredictProg = cl::Program(m_clContext, clPredictSrc);

  // Generic feature trick
  /** \todo Store feature type typedef file in a platform indipendent path */
  std::ofstream clFeatTypeFile("/tmp/feat_type.cl");
  std::string code;
  FeatTypeTrait<FeatType>::getCLTypedefCode(code);
  clFeatTypeFile.write(code.c_str(), code.length());
  clFeatTypeFile.close();

  // General image type trick:
  // - if the image contains up to 4 channel, work with image2d_t;
  // - if the image has more than 4 channel, work with 3D images (i.e. image3d_t)
  std::ofstream clImgTypeFile("/tmp/image_type.cl");
  std::string imgTypedefCodeHeader("#ifndef __IMG_TYPE\n#define __IMG_TYPE\n\n");
  std::string imgTypedefCode((nChannels<=4) ?
			     "typedef image2d_t image_t;\n" : 
			     "typedef image3d_t image_t;\n");
  std::string imgTypedefCodeFooter("\n#endif //__IMG_TYPE");
  clImgTypeFile.write(imgTypedefCodeHeader.c_str(), imgTypedefCodeHeader.length());
  clImgTypeFile.write(imgTypedefCode.c_str(), imgTypedefCode.length());
  clImgTypeFile.write(imgTypedefCodeFooter.c_str(), imgTypedefCodeFooter.length());
  clImgTypeFile.close();

  std::stringstream opts;
  opts << "-I/tmp -Ikernels -I" << featureKernelPath;
  
  try
  {
    m_clPredictProg.build(opts.str().c_str());
  }
  catch (cl::Error e)
  {
    std::string buildLog;
    m_clPredictProg.getBuildInfo(m_clDevice, CL_PROGRAM_BUILD_LOG, &buildLog);
    
    std::cerr << buildLog << std::endl;
    throw buildLog;
  }

  m_clPredictKern = cl::Kernel(m_clPredictProg, "predict");
  m_clComputePosteriorKern = cl::Kernel(m_clPredictProg, "computePosterior");

  // Init OpenCL image objects used for prediction
  // Note: use a large image size at beginning (e.g. 4096x4096). If a wider image
  // must be processed, resize all buffer
  // Note: using a large buffer allows work-items to access outside real image bounds
  // during kernel computation. This must be prevented in some way...
  // Note: we used pinned-memory trick for images/buffers that are read/written by the host
  m_internalImgWidth = 2048;
  m_internalImgHeight = 2048;
  _initImgObjects(m_internalImgWidth, m_internalImgHeight, false);

  // Done
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::~CLClassifier()
{
  m_clQueue.enqueueUnmapMemObject(m_clPosteriorPinn, m_clPosteriorPinnPtr);
  m_clQueue.enqueueUnmapMemObject(m_clPredictImgPinn, m_clPredictImgPinnPtr);
  m_clQueue.enqueueUnmapMemObject(m_clMaskPinn, m_clMaskPinnPtr);
  m_clQueue.enqueueUnmapMemObject(m_clImgPinn, m_clImgPinnPtr);
  delete m_clImg; 
}



template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>&
CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::operator<<(
  const Tree<FeatType, FeatDim, nClasses>& tree)
{
  unsigned int treeDepth = tree.getDepth();
  unsigned int nNodes = (2<<(treeDepth-1))-1;

  m_clTreeLeftChildBuff.push_back(cl::Buffer(m_clContext,
					     CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
					     nNodes*sizeof(cl_uint),
					     (void*)tree.getLeftChildren()));
  m_clTreeFeaturesBuff.push_back(cl::Buffer(m_clContext,
					    CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
					    nNodes*sizeof(FeatType)*FeatDim,
					    (void*)tree.getFeatures()));
  m_clTreeThrsBuff.push_back(cl::Buffer(m_clContext,
					CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
					nNodes*sizeof(FeatType),
					(void*)tree.getThresholds()));
  m_clTreePosteriorsBuff.push_back(cl::Buffer(m_clContext,
					      CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
					      nNodes*sizeof(cl_float)*nClasses,
					      (void*)tree.getPosteriors()));
  m_treeDepth.push_back(tree.getDepth());
  m_nTrees++;

  return *this;
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::predict(
  unsigned int treeID, const Image<ImgType, nChannels> &image,
  Image<int, 1> &prediction)
{
  Image<unsigned char, 1> mask(image.getWidth(), image.getHeight());
  std::fill_n(mask.getData(), mask.getWidth()*mask.getHeight(), 255);
  predict(treeID, image, prediction, mask);
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::predict(
  unsigned int treeID, const Image<ImgType, nChannels> &image,
  Image<int, 1> &prediction,
  Image<unsigned char, 1> &mask)
{
  cl::size_t<3> origin, region;
  size_t fillWidth, fillHeight;
  
  fillWidth = (image.getWidth()%WG_WIDTH) ? WG_WIDTH-(image.getWidth()%WG_WIDTH) : 0;
  fillHeight = (image.getHeight()%WG_HEIGHT) ? WG_HEIGHT-(image.getHeight()%WG_HEIGHT) : 0;

  if (image.getWidth()+fillWidth > m_internalImgWidth ||
      image.getHeight()+fillHeight > m_internalImgHeight)
  {
    m_internalImgWidth = image.getWidth()+fillWidth;
    m_internalImgHeight = image.getHeight()+fillHeight;
    _initImgObjects(m_internalImgWidth, m_internalImgHeight, true);
  }

  origin[0]=0; origin[1]=0; origin[2]=0;
  region[0]=image.getWidth()+fillWidth; region[1]=image.getHeight()+fillHeight;
  region[2] = 1;

  //cl::ImageFormat clImgFormat;
  //ImgTypeTrait<ImgType, nChannels>::toCLImgFmt(clImgFormat);
  //if (nChannels<=4)
  //{
  //  m_clImg = new cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			      region[0], region[1]);
  //}
  //else
  //{
  //  m_clImg = new cl::Image3D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			      region[0], region[1], nChannels);
  //}

  //clImgFormat.image_channel_order = CL_R;
  //clImgFormat.image_channel_data_type = CL_SIGNED_INT32;
  //region[2]=1;
  //m_clNodesIDImg = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			       region[0], region[1]);
  //m_clPredictImg = cl::Image2D(m_clContext, CL_MEM_WRITE_ONLY, clImgFormat,
  //			       region[0], region[1]);
  //
  //clImgFormat.image_channel_data_type = CL_UNSIGNED_INT8;
  //m_clMask = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			 region[0], region[1]);

  // Init prediction for depth 1 (i.e. all the pixels start from 0 (root) node)
#ifdef CL_VERSION_1_2
  cl_int4 fillColor = {0, 0, 0, 0};
  m_clQueue.enqueueFillImage(m_clNodesIDImg, fillColor, origin, region);
#else
  size_t rowPitch;
  char *tmpImgPtr = (char*)m_clQueue.enqueueMapImage(m_clNodesIDImg, CL_TRUE, CL_MAP_WRITE,
						     origin, region, &rowPitch, NULL);
  std::fill_n(tmpImgPtr, rowPitch*region[1], 0);
  m_clQueue.enqueueUnmapMemObject(m_clNodesIDImg, tmpImgPtr);
#endif
  
  // Load current image and mask
  region[0]=image.getWidth(); region[1]=image.getHeight();
  region[2]= (nChannels<=4) ? 1 : nChannels;
  std::copy(image.getData(), image.getData()+region[0]*region[1]*nChannels,
	    m_clImgPinnPtr);
  if (nChannels<=4)
  {
    //m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image2D*>(m_clImg),
    //				CL_FALSE, origin, region, 0, 0, (void*)image.getData());
    m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image2D*>(m_clImg),
				CL_FALSE, origin, region, 0, 0,
				(void*)m_clImgPinnPtr);
  }
  else
  {
    //m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image3D*>(m_clImg),
    //				CL_FALSE, origin, region, 0, 0, (void*)image.getData());
    m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image3D*>(m_clImg),
				CL_FALSE, origin, region, 0, 0,
				(void*)m_clImgPinnPtr);
  }
 
  region[2]=1;
  std::copy(mask.getData(), mask.getData()+region[0]*region[1], m_clMaskPinnPtr);
  //m_clQueue.enqueueWriteImage(m_clMask, CL_FALSE, origin, region, 0, 0, (void*)mask.getData());
  m_clQueue.enqueueWriteImage(m_clMask, CL_FALSE, origin, region, 0, 0, (void*)m_clMaskPinnPtr);

  // Set parameters and start prediction
  if (nChannels<=4)
  {
    m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image2D*>(m_clImg));
  }
  else
  {
    m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image3D*>(m_clImg));
  }
  m_clPredictKern.setArg(1, m_clMask);
  m_clPredictKern.setArg(2, nChannels);
  m_clPredictKern.setArg(3, image.getWidth());
  m_clPredictKern.setArg(4, image.getHeight());
  m_clPredictKern.setArg(5, m_clTreeLeftChildBuff.at(treeID));
  m_clPredictKern.setArg(6, m_clTreeFeaturesBuff.at(treeID));
  m_clPredictKern.setArg(7, FeatDim);
  m_clPredictKern.setArg(8, m_clTreeThrsBuff.at(treeID));
  m_clPredictKern.setArg(9, m_clTreePosteriorsBuff.at(treeID));
  m_clPredictKern.setArg(10, m_clNodesIDImg);
  m_clPredictKern.setArg(11, m_clPredictImg);
  m_clPredictKern.setArg(12, cl::Local(sizeof(FeatType)*WG_WIDTH*WG_HEIGHT*FeatDim));

  for (int d=1; d<m_treeDepth.at(treeID); d++)
  {
    m_clQueue.enqueueNDRangeKernel(m_clPredictKern,
				   cl::NullRange,
				   cl::NDRange(image.getWidth()+fillWidth,
					       image.getHeight()+fillHeight),
				   cl::NDRange(WG_WIDTH, WG_HEIGHT));
    if (d<m_treeDepth.at(treeID)-1)
      m_clQueue.enqueueCopyImage(m_clPredictImg, m_clNodesIDImg, origin, origin, region);
  }

  // Read results
  //m_clQueue.enqueueReadImage(m_clPredictImg, CL_TRUE, origin, region, 0, 0, (void*)prediction.getData());
  m_clQueue.enqueueReadImage(m_clPredictImg, CL_TRUE, origin, region, 0, 0, 
			     (void*)m_clPredictImgPinnPtr);
  std::copy(m_clPredictImgPinnPtr, m_clPredictImgPinnPtr+region[0]*region[1],
	    prediction.getData());

  // Done
  //delete m_clImg;
}

template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::predict(
  const Image<ImgType, nChannels> &image,
  Image<float, nClasses> &prediction)
{
  Image<unsigned char, 1> mask(image.getWidth(), image.getHeight());
  std::fill_n(mask.getData(), mask.getWidth()*mask.getHeight(), 255);
  predict(image, prediction, mask);
}

template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::predict(
  const Image<ImgType, nChannels> &image,
  Image<float, nClasses> &posterior,
  Image<unsigned char, 1> &mask)
{
  cl::size_t<3> origin, region;
  size_t fillWidth, fillHeight;
  
  fillWidth = (image.getWidth()%WG_WIDTH) ? WG_WIDTH-(image.getWidth()%WG_WIDTH) : 0;
  fillHeight = (image.getHeight()%WG_HEIGHT) ? WG_HEIGHT-(image.getHeight()%WG_HEIGHT) : 0;

  if (image.getWidth()+fillWidth > m_internalImgWidth ||
      image.getHeight()+fillHeight > m_internalImgHeight)
  {
    m_internalImgWidth = image.getWidth()+fillWidth;
    m_internalImgHeight = image.getHeight()+fillHeight;
    _initImgObjects(m_internalImgWidth, m_internalImgHeight, true);
  }

  origin[0]=0; origin[1]=0; origin[2]=0;
  region[0]=image.getWidth()+fillWidth; region[1]=image.getHeight()+fillHeight;
  region[2]= (nChannels<=4) ? 1 : nChannels;

  //cl::ImageFormat clImgFormat;
  //ImgTypeTrait<ImgType, nChannels>::toCLImgFmt(clImgFormat);
  //if (nChannels<=4)
  //{
  //  m_clImg = new cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			      region[0], region[1]);
  //}
  //else
  //{
  // m_clImg = new cl::Image3D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			      region[0], region[1], nChannels);
  //}

  //clImgFormat.image_channel_order = CL_R;
  //clImgFormat.image_channel_data_type = CL_SIGNED_INT32;
  //region[2]=1;
  //m_clNodesIDImg = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			       region[0], region[1]);

  //m_clPredictImg = cl::Image2D(m_clContext, CL_MEM_READ_WRITE, clImgFormat,
  //			       region[0], region[1]);

  //clImgFormat.image_channel_data_type = CL_UNSIGNED_INT8;
  //m_clMask = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
  //			 region[0], region[1]);

  //m_clPosteriorBuff = cl::Buffer(m_clContext, CL_MEM_WRITE_ONLY,
  //				 image.getWidth()*image.getHeight()*nClasses*sizeof(cl_float),
  //				 NULL);

  // Init posterior images channels to zero
#ifdef CL_VERSION_1_2
  cl_float zeroPosterior = 0.f;
  m_clQueue.enqueueFillBuffer<cl_float>(m_clPosteriorBuff, zeroPosterior,
					0, image.getWidth()*image.getHeight()*nClasses*sizeof(cl_float));
#else
  char *tmpBuffPtr = (char*)m_clQueue.enqueueMapBuffer(m_clPosteriorBuff, CL_TRUE, CL_MAP_WRITE,
                                                       0, image.getWidth()*image.getHeight()*nClasses*sizeof(cl_float));
  std::fill_n(tmpBuffPtr, image.getWidth()*image.getHeight()*nClasses*sizeof(cl_float), 0);
  m_clQueue.enqueueUnmapMemObject(m_clPosteriorBuff, tmpBuffPtr);
#endif
  std::fill_n(posterior.getData(), posterior.getWidth()*posterior.getHeight()*nClasses, 0.f);
  
  // Load current image and mask
  region[0]=image.getWidth(); region[1]=image.getHeight();
  region[2]= (nChannels<=4) ? 1 : nChannels;
  std::copy(image.getData(), image.getData()+region[0]*region[1]*nChannels,
	    m_clImgPinnPtr);
  if (nChannels<=4)
  {
    //m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image2D*>(m_clImg),
    //				CL_FALSE, origin, region, 0, 0, (void*)image.getData());
    m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image2D*>(m_clImg),
				CL_FALSE, origin, region, 0, 0,
				(void*)m_clImgPinnPtr);
  }
  else
  {
    //m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image3D*>(m_clImg),
    //				CL_FALSE, origin, region, 0, 0, (void*)image.getData());
    m_clQueue.enqueueWriteImage(*reinterpret_cast<cl::Image3D*>(m_clImg),
				CL_FALSE, origin, region, 0, 0,
				(void*)m_clImgPinnPtr);
  }

  region[2]=1;
  std::copy(mask.getData(), mask.getData()+region[0]*region[1], m_clMaskPinnPtr);
  //m_clQueue.enqueueWriteImage(m_clMask, CL_FALSE, origin, region, 0, 0, (void*)mask.getData());
  m_clQueue.enqueueWriteImage(m_clMask, CL_FALSE, origin, region, 0, 0, (void*)m_clMaskPinnPtr);
  
  // Init posterior image (stores leaves posterior for current tree)
  //float *currPosteriorBuff = new float[image.getWidth()*image.getHeight()*nClasses];

  // Set kernel arguments that do not change between calls
  if (nChannels<=4)
  {
    m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image2D*>(m_clImg));
  }
  else
  {
    m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image3D*>(m_clImg));
  }
  m_clPredictKern.setArg(1, m_clMask);
  m_clPredictKern.setArg(2, nChannels);
  m_clPredictKern.setArg(3, image.getWidth());
  m_clPredictKern.setArg(4, image.getHeight());
  m_clPredictKern.setArg(7, FeatDim);
  m_clPredictKern.setArg(10, m_clNodesIDImg);
  m_clPredictKern.setArg(11, m_clPredictImg);
  m_clPredictKern.setArg(12, cl::Local(sizeof(FeatType)*WG_WIDTH*WG_HEIGHT*FeatDim));

  m_clComputePosteriorKern.setArg(0, m_clPredictImg);
  m_clComputePosteriorKern.setArg(1, image.getWidth());
  m_clComputePosteriorKern.setArg(2, image.getHeight());
  m_clComputePosteriorKern.setArg(3, nClasses);
  m_clComputePosteriorKern.setArg(5, m_clMask);
  m_clComputePosteriorKern.setArg(6, m_clPosteriorBuff);
  
  
  for (int t=0; t<m_nTrees; t++)
  {
    // Init prediction for depth 1 (i.e. all the pixels start from 0 (root) node)
    region[0]=image.getWidth()+fillWidth;
    region[1]=image.getHeight()+fillHeight;
    region[2]=1;
#ifdef CL_VERSION_1_2
    cl_int4 fillColor = {0, 0, 0, 0};
    m_clQueue.enqueueFillImage(m_clNodesIDImg, fillColor, origin, region);
#else
    size_t rowPitch;
    char *tmpImgPtr = (char*)m_clQueue.enqueueMapImage(m_clNodesIDImg, CL_TRUE, CL_MAP_WRITE,
						       origin, region, &rowPitch, NULL);
    std::fill_n(tmpImgPtr, rowPitch*region[1], 0);
    m_clQueue.enqueueUnmapMemObject(m_clNodesIDImg, tmpImgPtr);
#endif

    m_clPredictKern.setArg(5, m_clTreeLeftChildBuff.at(t));
    m_clPredictKern.setArg(6, m_clTreeFeaturesBuff.at(t));
    m_clPredictKern.setArg(8, m_clTreeThrsBuff.at(t));
    m_clPredictKern.setArg(9, m_clTreePosteriorsBuff.at(t));

    m_clComputePosteriorKern.setArg(4, m_clTreePosteriorsBuff.at(t));

    
    for (int d=1; d<m_treeDepth.at(t); d++)
    {
      m_clQueue.enqueueNDRangeKernel(m_clPredictKern,
				     cl::NullRange,
				     cl::NDRange(image.getWidth()+fillWidth,
						 image.getHeight()+fillHeight),
				     cl::NDRange(WG_WIDTH, WG_HEIGHT));
      if (d<m_treeDepth.at(t)-1)
	m_clQueue.enqueueCopyImage(m_clPredictImg, m_clNodesIDImg, origin, origin, region);
    }

    
    m_clQueue.enqueueNDRangeKernel(m_clComputePosteriorKern,
				   cl::NullRange,
				   cl::NDRange(image.getWidth()+fillWidth,
					       image.getHeight()+fillHeight),
				   cl::NDRange(WG_WIDTH, WG_HEIGHT));

    // Read results
    //m_clQueue.enqueueReadBuffer(m_clPosteriorBuff, CL_TRUE,
    //                          0, image.getWidth()*image.getHeight()*nClasses*sizeof(cl_float),
    //				(void*)currPosteriorBuff);
    m_clQueue.enqueueReadBuffer(m_clPosteriorBuff, CL_TRUE,
				0, image.getWidth()*image.getHeight()*nClasses*sizeof(cl_float),
				(void*)m_clPosteriorPinnPtr);
    

    // Sum current posterior to total posterior
    /** \todo Optimize with intrinsics */
    //#pragma omp parallel for
    for (int l=0; l<nClasses; l++)
    {
      float *posteriorPtr = posterior.getData()+l*(image.getWidth()*image.getHeight());
      //float *currPosteriorPtr = currPosteriorBuff+l*(image.getWidth()*image.getHeight());
      float *currPosteriorPtr = m_clPosteriorPinnPtr+l*(image.getWidth()*image.getHeight());
      unsigned char *maskPtr = mask.getData();
      for (int v=0; v<image.getHeight(); v++)
      {
	for (int u=0; u<image.getWidth(); u++, posteriorPtr++, currPosteriorPtr++, maskPtr++)
	{
          if (!(*maskPtr)) continue;
	  *posteriorPtr += *currPosteriorPtr;
	}
      }
    }
  }

  // Normalize posterior
  for (int i=0; i<image.getWidth()*image.getHeight()*nClasses; i++) 
    posterior.getData()[i] /= m_nTrees;

  // Done
  //delete m_clImg;
  //delete []currPosteriorBuff;
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLClassifier<ImgType, nChannels, FeatType, FeatDim, nClasses>::_initImgObjects(size_t width, size_t height, bool deallocate)
{
  if (deallocate)
  {
    m_clQueue.enqueueUnmapMemObject(m_clPosteriorPinn, m_clPosteriorPinnPtr);
    m_clQueue.enqueueUnmapMemObject(m_clPredictImgPinn, m_clPredictImgPinnPtr);
    m_clQueue.enqueueUnmapMemObject(m_clMaskPinn, m_clMaskPinnPtr);
    m_clQueue.enqueueUnmapMemObject(m_clImgPinn, m_clImgPinnPtr);
    delete m_clImg;
  }


  cl::size_t<3> origin, region;
  
  origin[0]=0; origin[1]=0; origin[2]=0;
  region[0]=width; region[1]=height;
  region[2]= (nChannels<=4) ? 1 : nChannels;


  // Init the memory objects for the input image
  cl::ImageFormat clImgFormat;
  ImgTypeTrait<ImgType, nChannels>::toCLImgFmt(clImgFormat);
  if (nChannels<=4)
  {
    m_clImg = new cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
			      region[0], region[1]);
  }
  else
  {
    m_clImg = new cl::Image3D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
			      region[0], region[1], nChannels);
  }
  m_clImgPinn = cl::Buffer(m_clContext,
			   CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
			   region[0]*region[1]*nChannels*sizeof(ImgType));
  m_clImgPinnPtr = 
    reinterpret_cast<ImgType*>(m_clQueue.enqueueMapBuffer(m_clImgPinn, CL_TRUE,
							  CL_MAP_WRITE,
							  0, region[0]*region[1]*nChannels*sizeof(ImgType)));


  // Init memory objects for prediction results
  clImgFormat.image_channel_order = CL_R;
  clImgFormat.image_channel_data_type = CL_SIGNED_INT32;
  region[2]=1;
  m_clNodesIDImg = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
			       region[0], region[1]);

  m_clPredictImg = cl::Image2D(m_clContext, CL_MEM_READ_WRITE, clImgFormat,
			       region[0], region[1]);
  m_clPredictImgPinn = cl::Buffer(m_clContext,
				  CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
				  region[0]*region[1]*sizeof(int));
  m_clPredictImgPinnPtr =
    reinterpret_cast<int*>(m_clQueue.enqueueMapBuffer(m_clPredictImgPinn, CL_TRUE,
						      CL_MAP_WRITE|CL_MAP_WRITE,
						      0, region[0]*region[1]*sizeof(int)));

  // Init memory objects for mask
  clImgFormat.image_channel_data_type = CL_UNSIGNED_INT8;
  m_clMask = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clImgFormat,
			 region[0], region[1]);
  m_clMaskPinn = cl::Buffer(m_clContext,
			    CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
			    region[0]*region[1]*sizeof(unsigned char));
  m_clMaskPinnPtr =
    reinterpret_cast<unsigned char*>(m_clQueue.enqueueMapBuffer(m_clMaskPinn, CL_TRUE,
								CL_MAP_WRITE,
								0, region[0]*region[1]*sizeof(unsigned char)));
 
  // Init memory objects for posterior
  m_clPosteriorBuff = cl::Buffer(m_clContext, CL_MEM_WRITE_ONLY,
				 region[0]*region[1]*nClasses*sizeof(cl_float),
				 NULL);
  m_clPosteriorPinn = cl::Buffer(m_clContext,
				 CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,
				 region[0]*region[1]*nClasses*sizeof(cl_float));
  m_clPosteriorPinnPtr =
    reinterpret_cast<float*>(m_clQueue.enqueueMapBuffer(m_clPosteriorPinn, CL_TRUE,
							CL_MAP_READ,
							0, region[0]*region[1]*nClasses*sizeof(cl_float)));
  
  // Done
}
