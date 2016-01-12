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
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::_initTrain(
  Tree<FeatType, FeatDim, nClasses> &tree,
  const TrainingSet<ImgType, nChannels> &trainingSet,
  const TreeTrainerParameters<FeatType, FeatDim> &params,
  unsigned int startDepth, unsigned int endDepth)
{
  unsigned int nNodes = (2<<(endDepth-1))-1;
  cl_int errCode;

  // Init OpenCL tree buffers and load corresponding data
  m_clTreeLeftChildBuff = cl::Buffer(m_clContext,
				     CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				     nNodes*sizeof(cl_uint),
				     (void*)tree.getLeftChildren());
  m_clTreeFeaturesBuff = cl::Buffer(m_clContext,
				    CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				    nNodes*sizeof(FeatType)*FeatDim,
				    (void*)tree.getFeatures());
  m_clTreeThrsBuff = cl::Buffer(m_clContext,
				CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				nNodes*sizeof(FeatType),
				(void*)tree.getThresholds());
  m_clTreePosteriorsBuff = cl::Buffer(m_clContext,
				      CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				      nNodes*sizeof(cl_float)*nClasses,
				      (void*)tree.getPosteriors());

  // Init per-node total and per-class number of samples
  m_perNodeTotSamples = new unsigned int[nNodes];
  m_perClassTotSamples = new unsigned int[nNodes*nClasses];
  std::fill_n(m_perNodeTotSamples, nNodes, 0);
  std::fill_n(m_perClassTotSamples, nNodes*nClasses, 0);

  // Init to-skip flags for training set images
  m_toSkipTsImg = new bool[trainingSet.getImages().size()];
  m_skippedTsImg = new bool[trainingSet.getImages().size()];
  std::fill_n(m_skippedTsImg, trainingSet.getImages().size(), false);

  // Init OpenCL training set image buffer:
  // - first of all, iterate through the training set and find the maximum
  //   image width/height
  m_maxTsImgWidth=0;
  m_maxTsImgHeight=0;
  m_maxTsImgSamples=0;
  m_perNodeTotSamples[0] = 0;
  const std::vector<TrainingSetImage<ImgType, nChannels> > &tsImages = trainingSet.getImages();
  for (typename std::vector<TrainingSetImage<ImgType, nChannels> >::const_iterator it=tsImages.begin();
       it!=tsImages.end(); ++it)
  {
    const TrainingSetImage<ImgType, nChannels> &currImage=*it;
    
    if (currImage.getWidth()>m_maxTsImgWidth) m_maxTsImgWidth=currImage.getWidth();
    if (currImage.getHeight()>m_maxTsImgHeight) m_maxTsImgHeight=currImage.getHeight();

    // Get maximum number of sampled pixels as well
    if (currImage.getNSamples()>m_maxTsImgSamples) m_maxTsImgSamples=currImage.getNSamples();

    /** \todo update here total number of pixel per class at root node */
    m_perNodeTotSamples[0]+=currImage.getNSamples();
  }

  // Make the maximum width and height a multiple of the, respectively, work-group x and y
  // dimension
  //m_maxTsImgWidth += (m_maxTsImgWidth%WG_WIDTH) ? WG_WIDTH-(m_maxTsImgWidth%WG_WIDTH) : 0;
  //m_maxTsImgHeight += (m_maxTsImgHeight%WG_HEIGHT) ? WG_HEIGHT-(m_maxTsImgHeight%WG_HEIGHT) : 0;
  m_maxTsImgWidth += (m_maxTsImgWidth%16) ? 16-(m_maxTsImgWidth%16) : 0;
  m_maxTsImgHeight += (m_maxTsImgHeight%16) ? 16-(m_maxTsImgHeight%16) : 0;

  // - initialize OpenCL images
  cl::size_t<3> origin, region;
  size_t rowPitch;
  origin[0]=0; origin[1]=0; origin[2]=0;
  region[0]=m_maxTsImgWidth; region[1]=m_maxTsImgHeight;
  region[2]= (nChannels<=4) ? 1 : nChannels;

  cl::ImageFormat clTsImgFormat;
  ImgTypeTrait<ImgType, nChannels>::toCLImgFmt(clTsImgFormat);
  if (nChannels<=4)
  {
    m_clTsImg1 = new cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight);
    m_clTsImg2 = new cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight);
  }
  else
  {
    m_clTsImg1 = new cl::Image3D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight, nChannels);
    m_clTsImg2 = new cl::Image3D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight, nChannels);
  }
  m_clTsImgPinn = cl::Buffer(m_clContext,
			     CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
			     m_maxTsImgWidth*m_maxTsImgHeight*nChannels*sizeof(ImgType)*2);
  m_clTsImgPinnPtr = 
    reinterpret_cast<ImgType*>(m_clQueue1.enqueueMapBuffer(m_clTsImgPinn, CL_TRUE,
							   CL_MAP_WRITE,
							   0, m_maxTsImgWidth*m_maxTsImgHeight*nChannels*sizeof(ImgType)*2));

  clTsImgFormat.image_channel_order = CL_R;
  clTsImgFormat.image_channel_data_type = CL_UNSIGNED_INT8;
  region[2] = 1;
  m_clTsLabelsImg1 = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight);
  m_clTsLabelsImg2 = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight);
  m_clTsLabelsImgPinn = cl::Buffer(m_clContext,
				   CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
				   m_maxTsImgWidth*m_maxTsImgHeight*sizeof(cl_uchar)*2);
  m_clTsLabelsImgPinnPtr = 
    reinterpret_cast<unsigned char*>(m_clQueue1.enqueueMapBuffer(m_clTsLabelsImgPinn, CL_TRUE,
								 CL_MAP_WRITE,
								 0, m_maxTsImgWidth*m_maxTsImgHeight*sizeof(cl_uchar)*2));

  clTsImgFormat.image_channel_data_type = CL_SIGNED_INT32;
  m_clTsNodesIDImg1 = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				  m_maxTsImgWidth, m_maxTsImgHeight);
  m_clTsNodesIDImg2 = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				  m_maxTsImgWidth, m_maxTsImgHeight);
  m_clTsNodesIDImgPinn = cl::Buffer(m_clContext,
				     CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
				     m_maxTsImgWidth*m_maxTsImgHeight*sizeof(cl_uint)*GLOBAL_HISTOGRAM_FIFO_SIZE);
  m_clTsNodesIDImgPinnPtr =
    reinterpret_cast<int*>(m_clQueue1.enqueueMapBuffer(m_clTsNodesIDImgPinn, CL_TRUE,
						       CL_MAP_READ|CL_MAP_WRITE,
						       0, m_maxTsImgWidth*m_maxTsImgHeight*sizeof(cl_uint)*GLOBAL_HISTOGRAM_FIFO_SIZE));

  m_clPredictImg1 = cl::Image2D(m_clContext, CL_MEM_WRITE_ONLY, clTsImgFormat,
				m_maxTsImgWidth, m_maxTsImgHeight);
  m_clPredictImg2 = cl::Image2D(m_clContext, CL_MEM_WRITE_ONLY, clTsImgFormat,
				m_maxTsImgWidth, m_maxTsImgHeight);
  
  // Init OpenCL buffers for per-image histogram computation
  FeatType *tmpFeatLowBounds = new FeatType[FeatDim];
  FeatType *tmpFeatUpBounds = new FeatType[FeatDim];
  std::copy(params.featLowBounds, params.featLowBounds+FeatDim, tmpFeatLowBounds);
  std::copy(params.featUpBounds, params.featUpBounds+FeatDim, tmpFeatUpBounds);
  m_clFeatLowBoundsBuff = cl::Buffer(m_clContext,
				     CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				     FeatDim*sizeof(FeatType),
				     (void*)tmpFeatLowBounds);
  m_clFeatUpBoundsBuff = cl::Buffer(m_clContext,
				    CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				    FeatDim*sizeof(FeatType),
				    (void*)tmpFeatUpBounds);

  m_clTsSamplesBuff1 = cl::Buffer(m_clContext,
				  CL_MEM_READ_ONLY,
				  m_maxTsImgSamples*sizeof(cl_uint));
  m_clTsSamplesBuff2 = cl::Buffer(m_clContext,
				  CL_MEM_READ_ONLY,
				  m_maxTsImgSamples*sizeof(cl_uint));
  m_clTsSamplesBuffPinn = cl::Buffer(m_clContext,
				     CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
				     m_maxTsImgSamples*sizeof(cl_uint)*2);
  m_clTsSamplesBuffPinnPtr =
    reinterpret_cast<unsigned int*>(m_clQueue1.enqueueMapBuffer(m_clTsSamplesBuffPinn, CL_TRUE,
								CL_MAP_WRITE,
								0, m_maxTsImgSamples*sizeof(cl_uint)*2));

  
  // Note:
  // - 4D Historam (sample-ID, feature, class, threshold) can be compressed to 3D since we can access
  //   the sample class from labels image
  size_t perImgHistogramSize = m_maxTsImgSamples*params.nFeatures*params.nThresholds;
  m_clPerImgHistBuff1 = cl::Buffer(m_clContext,
				   CL_MEM_WRITE_ONLY,
				   perImgHistogramSize*sizeof(cl_uchar));
  m_clPerImgHistBuff2 = cl::Buffer(m_clContext,
				   CL_MEM_WRITE_ONLY,
				   perImgHistogramSize*sizeof(cl_uchar));
  m_clPerImgHistBuffPinn = cl::Buffer(m_clContext,
				      CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR,
				      perImgHistogramSize*sizeof(cl_uchar)*GLOBAL_HISTOGRAM_FIFO_SIZE);
  m_clPerImgHistBuffPinnPtr =
    reinterpret_cast<unsigned char*>(m_clQueue1.enqueueMapBuffer(m_clPerImgHistBuffPinn, CL_TRUE,
								 CL_MAP_READ,
								 0, perImgHistogramSize*sizeof(cl_uchar)*GLOBAL_HISTOGRAM_FIFO_SIZE));


  // Init buffers used for best per-node feature/threshold pair learning
  /** 
   * \todo how to find a proper value for per-thread feature/threshold pairs and parallely-learnt
   *       node's best feature/threshold pair? At least avoid hard-coded values and parameterize them.
   * \todo check if per-thread pairs number is a multiple of total per-node pairs
  */
  size_t maxFrontierSize = (endDepth>2) ? (2<<(endDepth-3)) : 1;
  size_t perNodeHistogramSize = nClasses*params.nFeatures*params.nThresholds;
  unsigned int perThreadFeatThrPairs = PER_THREAD_FEAT_THR_PAIRS;
  unsigned int parLearntNodes = (maxFrontierSize>PARALLEL_LEARNT_NODES) ? 
    PARALLEL_LEARNT_NODES : maxFrontierSize;
  size_t learnBuffsSize = parLearntNodes*(params.nFeatures*params.nThresholds)/perThreadFeatThrPairs;
  m_clHistogramBuff = cl::Buffer(m_clContext,
				 CL_MEM_READ_ONLY,
				 parLearntNodes*perNodeHistogramSize*sizeof(cl_uint));
  m_clBestFeaturesBuff = cl::Buffer(m_clContext,
				    CL_MEM_WRITE_ONLY,
				    learnBuffsSize*sizeof(cl_uint));
  m_clBestThresholdsBuff = cl::Buffer(m_clContext,
				      CL_MEM_WRITE_ONLY,
				      learnBuffsSize*sizeof(cl_uint));
  m_clBestEntropiesBuff = cl::Buffer(m_clContext,
				     CL_MEM_WRITE_ONLY,
				     learnBuffsSize*sizeof(cl_float));
  m_clPerClassTotSamplesBuff = cl::Buffer(m_clContext,
					  CL_MEM_READ_ONLY,
					  parLearntNodes*nClasses*sizeof(cl_uint));
				    
  // Set kernels arguments that does not change between calls:
  // - prediction
  // Hack: use the labels image as mask, i.e. assume pixels with non-zero label
  //       are marked as foreground pixels
  //m_clPredictKern.setArg(0, m_clTsImg);
  //m_clPredictKern.setArg(1, m_clTsLabelsImg);
  m_clPredictKern.setArg(2, nChannels);
  m_clPredictKern.setArg(5, m_clTreeLeftChildBuff);
  m_clPredictKern.setArg(6, m_clTreeFeaturesBuff);
  m_clPredictKern.setArg(7, FeatDim);
  m_clPredictKern.setArg(8, m_clTreeThrsBuff);
  m_clPredictKern.setArg(9, m_clTreePosteriorsBuff);
  //m_clPredictKern.setArg(10, m_clTsNodesIDImg);
  //m_clPredictKern.setArg(11, m_clPredictImg);
  //m_clPredictKern.setArg(12, cl::Local(sizeof(FeatType)*WG_WIDTH*WG_HEIGHT*FeatDim));
  m_clPredictKern.setArg(12, cl::Local(sizeof(FeatType)*256*FeatDim));

  // - per-image histogram update
  //m_clPerImgHistKern.setArg(0, m_clTsImg);
  m_clPerImgHistKern.setArg(1, nChannels);
  //m_clPerImgHistKern.setArg(4, m_clTsLabelsImg);
  //m_clPerImgHistKern.setArg(5, m_clTsNodesIDImg);
  //m_clPerImgHistKern.setArg(6, m_clTsSamplesBuff);
  m_clPerImgHistKern.setArg(8, FeatDim);
  m_clPerImgHistKern.setArg(9, m_clFeatLowBoundsBuff);
  m_clPerImgHistKern.setArg(10, m_clFeatUpBoundsBuff);
  m_clPerImgHistKern.setArg(11, params.nThresholds);
  m_clPerImgHistKern.setArg(12, params.thrLowBound);
  m_clPerImgHistKern.setArg(13, params.thrUpBound);
  //m_clPerImgHistKern.setArg(14, m_clPerImgHistBuff);
  m_clPerImgHistKern.setArg(15, tree.getID());
  m_clPerImgHistKern.setArg(18, m_clTreeLeftChildBuff);
  m_clPerImgHistKern.setArg(19, m_clTreePosteriorsBuff);
  m_clPerImgHistKern.setArg(20, cl::Local(sizeof(FeatType)*8));
  //m_clPerImgHistKern.setArg(21, cl::Local(sizeof(FeatType)*WG_WIDTH*WG_HEIGHT*FeatDim));
  m_clPerImgHistKern.setArg(21, cl::Local(sizeof(FeatType)*256*FeatDim));

  // - node's best feature/threshold learning
  m_clLearnBestFeatKern.setArg(0, m_clHistogramBuff);
  m_clLearnBestFeatKern.setArg(1, m_clPerClassTotSamplesBuff);
  m_clLearnBestFeatKern.setArg(2, params.nFeatures);
  m_clLearnBestFeatKern.setArg(3, params.nThresholds);
  m_clLearnBestFeatKern.setArg(4, nClasses);
  m_clLearnBestFeatKern.setArg(5, perThreadFeatThrPairs);
  m_clLearnBestFeatKern.setArg(6, m_clBestFeaturesBuff);
  m_clLearnBestFeatKern.setArg(7, m_clBestThresholdsBuff);
  m_clLearnBestFeatKern.setArg(8, m_clBestEntropiesBuff);


  // Init corresponding host buffers
  /** \todo use mapping/unmapping to avoid device/host copy */
  //m_tsNodesIDImg = new int[m_maxTsImgWidth*m_maxTsImgHeight*GLOBAL_HISTOGRAM_FIFO_SIZE];
  //m_perImgHist = new unsigned char[perImgHistogramSize*GLOBAL_HISTOGRAM_FIFO_SIZE];
  m_bestFeatures = new unsigned int[learnBuffsSize];
  m_bestThresholds = new unsigned int[learnBuffsSize];
  m_bestEntropies = new float[learnBuffsSize];

  // Done with OpenCL initialization


  // Init the global histogram:
  // define the global histogram as a vector of per-node histograms. The total size of
  // the global histogram (defined as number of per-node histograms simultaneously kept)
  // is limited by the smaller between maxFrontierSize and
  // GLOBAL_HISTOGRAM_MAX_SIZE/perNodeHistogramSize
  m_histogramSize = std::min(maxFrontierSize,
			     (size_t)floorl((double)GLOBAL_HISTOGRAM_MAX_SIZE/(perNodeHistogramSize*sizeof(unsigned int))));
  m_histogram = new unsigned int*[m_histogramSize];
  for (int i=0; i<m_histogramSize; i++) m_histogram[i] = new unsigned int[perNodeHistogramSize];


  // Buffer used to track to-train nodes for each depth
  m_frontier = new int[maxFrontierSize];


  // Note: the histogram for the root node is equal to the training set priors
  if (startDepth==1)
  {
    const TreeNode<FeatType, FeatDim> &rootNode = tree.getNode(0); 
    std::copy(trainingSet.getPriors(), trainingSet.getPriors()+nClasses, rootNode.m_posterior);
  }


  // Finally, init the random seed for features and thresholds sampling
  /*
  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(0, (2<<30)-1);
  m_seed = dist(gen);
  m_clPerImgHistKern.setArg(22, m_seed);
  */

  delete []tmpFeatUpBounds;
  delete []tmpFeatLowBounds;

  // Done
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
unsigned int CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::_initFrontier(
  Tree<FeatType, FeatDim, nClasses> &tree,
  const TreeTrainerParameters<FeatType, FeatDim> &params, unsigned int currDepth)
{
  size_t currFrontierSize = currDepth>1 ? (2<<(currDepth-2)) : 1;
  unsigned int startNode = currFrontierSize-1;
  unsigned int toTrainNodes = 0;

  m_frontierIdxMap.clear();

  if (currDepth>1)
  {
    for (unsigned int i=0; i<currFrontierSize; i++)
    {
      const TreeNode<FeatType, FeatDim> &currNode = tree.getNode(startNode+i);
      if (*currNode.m_leftChild==-1 && m_perNodeTotSamples[startNode+i]>params.perLeafSamplesThr)
      {
	m_frontier[toTrainNodes]=startNode+i;
	m_frontierIdxMap[startNode+i] = toTrainNodes;
	toTrainNodes++;
      }
    }
  }
  else
  {
    // Note: when starting from depth 1, root node gets always trained
    m_frontier[0] = 0;
    m_frontierIdxMap[0] = 0;
    toTrainNodes = 1;
  }

  return toTrainNodes;
}



template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
unsigned int CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::_initHistogram(
  const TreeTrainerParameters<FeatType, FeatDim> &params)
{
  unsigned int frontierSize = m_frontierIdxMap.size();

  size_t perNodeHistogramSize = params.nFeatures*params.nThresholds*nClasses;
  unsigned int nSlices = ceill((double)frontierSize/m_histogramSize);

  return nSlices;
}

template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::_cleanTrain()
{
  // Release pinned memory objects
  m_clQueue1.enqueueUnmapMemObject(m_clTsImgPinn, m_clTsImgPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clTsLabelsImgPinn, m_clTsLabelsImgPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clTsNodesIDImgPinn, m_clTsNodesIDImgPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clTsSamplesBuffPinn, m_clTsSamplesBuffPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clPerImgHistBuffPinn, m_clPerImgHistBuffPinnPtr);


  // Delete data dinamically allocated for current tree training
  delete []m_perNodeTotSamples;
  delete []m_perClassTotSamples;
  delete []m_toSkipTsImg;
  delete []m_skippedTsImg;
  delete m_clTsImg1;
  delete m_clTsImg2;
  delete []m_bestFeatures;
  delete []m_bestThresholds;
  delete []m_bestEntropies;
  for (int i=0; i<m_histogramSize; i++)
  {
    delete []m_histogram[i];
    m_histogram[i] = NULL;
  }
  delete []m_histogram;
  delete []m_frontier;
}
