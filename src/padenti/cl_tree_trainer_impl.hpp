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
#include <climits>
#include <cmath>
#include <sstream>
#include <fstream>
#include <vector>
#include <iterator>
#include <utility>
#include <boost/chrono/chrono.hpp>
#include <boost/log/trivial.hpp>
#include <padenti/cl_tree_trainer.hpp>
#include <padenti/cl_img_fmt_traits.hpp>
#include <padenti/cl_feat_fmt_traits.hpp>
#include <padenti/prng.hpp>

// TODO: delete
#include <cstring>


#include <padenti/hist_update.cl.inc>
#include <padenti/predict.cl.inc>
#include <padenti/learn_best_feature.cl.inc>


// Note: maximum allowed global (i.e. per-depth) histogram size, currently set to 8GB (2^30 unsigned int)
/** \todo parameterize, e.g. using a documented "internal parameters" argument */
//#define GLOBAL_HISTOGRAM_MAX_SIZE (24lu*(2<<29))
#define GLOBAL_HISTOGRAM_MAX_SIZE (20lu*(2<<29))

/** \todo "automagically" compute this value or parameterize it */
#define PER_THREAD_FEAT_THR_PAIRS (64)

/** \todo "automagically" compute this value or parameterize it */
#define PARALLEL_LEARNT_NODES (8)

// Size of the fifo queue used to parallelize global histogram updates
/** \todo parameterize */
//#define GLOBAL_HISTOGRAM_FIFO_SIZE (8)
#define GLOBAL_HISTOGRAM_FIFO_SIZE (2)

// Workgroup size for prediction and local histogram update
/** \todo parameterize workgroup sizes */
#define WG_PREDICT_HEIGHT (16)
#define WG_PREDICT_WIDTH (16)
#define WG_LHIST_UPDATE_HEIGHT (1)
#define WG_LHIST_UPDATE_WIDTH (256)


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::CLTreeTrainer(const std::string &featureKernelPath,
									      bool useCPU)
{
  // Get a OpenCL context using the first platform with a device of the specified type
  /** \todo provide API for platform and devices quering */
  m_clContext = cl::Context(useCPU ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU);
  
  // Get the first device of the specified type found
  /** \todo provide API for devices selection */
  m_clDevice = m_clContext.getInfo<CL_CONTEXT_DEVICES>()[0];

  //m_clQueue = cl::CommandQueue(m_clContext, m_clDevice, 0);
  m_clQueue1 = cl::CommandQueue(m_clContext, m_clDevice, CL_QUEUE_PROFILING_ENABLE);
  m_clQueue2 = cl::CommandQueue(m_clContext, m_clDevice, CL_QUEUE_PROFILING_ENABLE);

  // Compile training specific kernels
  std::string clHistUpdateStr(reinterpret_cast<const char*>(const_cast<const unsigned char*>(hist_update_cl)),
			      hist_update_cl_len);
  std::string clPredictStr(reinterpret_cast<const char*>(const_cast<const unsigned char*>(predict_cl)),
			   predict_cl_len);
  std::string clLearnBestFeatStr(reinterpret_cast<const char*>(const_cast<const unsigned char*>(learn_best_feature_cl)),
				 learn_best_feature_cl_len);

  cl::Program::Sources clHistUpdateSrc(1, std::make_pair(clHistUpdateStr.c_str(),
							 clHistUpdateStr.length()+1));  
  cl::Program::Sources clPredictSrc(1, std::make_pair(clPredictStr.c_str(),
						      clPredictStr.length()+1));
  cl::Program::Sources clLearnBestFeatSrc(1, std::make_pair(clLearnBestFeatStr.c_str(),
							    clLearnBestFeatStr.length()+1));

  m_clHistUpdateProg = cl::Program(m_clContext, clHistUpdateSrc);
  m_clPredictProg = cl::Program(m_clContext, clPredictSrc);
  m_clLearnBestFeatProg = cl::Program(m_clContext, clLearnBestFeatSrc);
  

  // Generic feature type trick:
  // define the OpenCL feature type using a typedef depending on the template-specified
  // C feature type. Use type-traits to retrieve the typedef code and save it on a temporary
  // file included by other OpenCL source.
  // Note: only standard types are supported
  /** \todo Store feature type file file in a platform indipendent path */
  std::ofstream clFeatTypeFile("/tmp/feat_type.cl");
  std::string featTypedefCode;
  FeatTypeTrait<FeatType>::getCLTypedefCode(featTypedefCode);
  clFeatTypeFile.write(featTypedefCode.c_str(), featTypedefCode.length());
  clFeatTypeFile.close();

  // Do the same for the image type:
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

  /** \todo better error handling */
  std::stringstream opts;
  //opts << "-I/tmp -Ikernels -I" << featureKernelPath;
  opts << "-I/tmp -I" << featureKernelPath;

  try
  {
    m_clHistUpdateProg.build(opts.str().c_str());
  }
  catch (cl::Error e)
  {
    std::string buildLog;
    m_clHistUpdateProg.getBuildInfo(m_clDevice, CL_PROGRAM_BUILD_LOG, &buildLog);

    std::cerr << buildLog << std::endl;
    throw buildLog;
  }

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

  try
  {
    m_clLearnBestFeatProg.build(opts.str().c_str());
  }
  catch (cl::Error e)
  {
    std::string buildLog;
    m_clLearnBestFeatProg.getBuildInfo(m_clDevice, CL_PROGRAM_BUILD_LOG, &buildLog);

    std::cerr << buildLog << std::endl;
    throw buildLog;
  }

  /** \todo avoid kernels name hardcoding? */
  m_clPerImgHistKern = cl::Kernel(m_clHistUpdateProg, "computePerImageHistogram");
  m_clPredictKern = cl::Kernel(m_clPredictProg, "predict");
  m_clLearnBestFeatKern = cl::Kernel(m_clLearnBestFeatProg, "learnBestFeature");
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::train(
  Tree<FeatType, FeatDim, nClasses> &tree,
  const TrainingSet<ImgType, nChannels> &trainingSet,
  const TreeTrainerParameters<FeatType, FeatDim> &params,
  unsigned int startDepth, unsigned int endDepth)
{
  /** \todo support a starting depth different from 1 */
  if (startDepth!=1) throw "Starting depth must be equal to 1";

  _initTrain(tree, trainingSet, params, startDepth, endDepth);
  

  for (unsigned int currDepth=startDepth; currDepth<endDepth; currDepth++)
  {
    boost::chrono::steady_clock::time_point perLevelTrainStart = 
      boost::chrono::steady_clock::now(); 

    unsigned int frontierSize = _initFrontier(tree, params, currDepth);
    unsigned int nSlices = _initHistogram(params);

    
    if (nSlices>1)
    {
      BOOST_LOG_TRIVIAL(info) << "Maximum allowed global histogram size reached: split in "
			      << nSlices << " slices";
    }
    

    // Flag all images as to-be-skipped: the flag will be set to false if at least one
    // image pixel is processed
    std::fill_n(m_toSkipTsImg, trainingSet.getImages().size(), true);

    for (unsigned int i=0; i<nSlices; i++)
    {
      _traverseTrainingSet(trainingSet, params, currDepth, i);
      _learnBestFeatThr(tree, params, currDepth, i);
    }

    // Update skipped images flags
    std::copy(m_toSkipTsImg, m_toSkipTsImg+trainingSet.getImages().size(), m_skippedTsImg);
  

    boost::chrono::duration<double> perLevelTrainTime =
      boost::chrono::duration_cast<boost::chrono::duration<double> >(boost::chrono::steady_clock::now() - 
								   perLevelTrainStart);
    
    BOOST_LOG_TRIVIAL(info) << "Depth " << currDepth << " trained in "
			    << perLevelTrainTime.count() << " seconds";
    
  }

  _cleanTrain();
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::~CLTreeTrainer()
{}


#include <padenti/cl_tree_trainer_impl_init.hpp>
#include <padenti/cl_tree_trainer_impl_traverse_ts.hpp>
#include <padenti/cl_tree_trainer_impl_learn_best_featthr.hpp>
