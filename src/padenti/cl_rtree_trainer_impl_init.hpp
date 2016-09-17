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
	  unsigned int RDim>
void CLRTreeTrainer<ImgType, nChannels, FeatType, FeatDim, RDim>::_initTrain(
  RTree<FeatType, FeatDim, RDim> &tree,
  const TrainingSetR<ImgType, nChannels, RDim> &trainingSet,
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
  m_clTreeValuesBuff = cl::Buffer(m_clContext,
				  CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
				  nNodes*sizeof(cl_float)*RDim,
				  (void*)tree.getValues());

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
  const std::vector<TrainingSetRImage<ImgType, nChannels, RDim> > &tsImages = trainingSet.getImages();
  for (typename std::vector<TrainingSetRImage<ImgType, nChannels, RDim> >::const_iterator it=tsImages.begin();
       it!=tsImages.end(); ++it)
  {
    const TrainingSetRImage<ImgType, nChannels, RDim> &currImage=*it;
    
    if (currImage.getWidth()>m_maxTsImgWidth) m_maxTsImgWidth=currImage.getWidth();
    if (currImage.getHeight()>m_maxTsImgHeight) m_maxTsImgHeight=currImage.getHeight();

    // Get maximum number of sampled pixels as well
    if (currImage.getNSamples()>m_maxTsImgSamples) m_maxTsImgSamples=currImage.getNSamples();
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
  m_clTsMaskImg1 = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight);
  m_clTsMaskImg2 = cl::Image2D(m_clContext, CL_MEM_READ_ONLY, clTsImgFormat,
				 m_maxTsImgWidth, m_maxTsImgHeight);
  m_clTsMaskImgPinn = cl::Buffer(m_clContext,
				   CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
				   m_maxTsImgWidth*m_maxTsImgHeight*sizeof(cl_uchar)*2);
  m_clTsMaskImgPinnPtr = 
    reinterpret_cast<unsigned char*>(m_clQueue1.enqueueMapBuffer(m_clTsMaskImgPinn, CL_TRUE,
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
  size_t perImgHistogramSize = m_maxTsImgSamples*params.nFeatures*params.nThresholds/8;
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
  /*
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
  */

				    
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
  m_clPredictKern.setArg(9, m_clTreeValuesBuff);
  //m_clPredictKern.setArg(10, m_clTsNodesIDImg);
  //m_clPredictKern.setArg(11, m_clPredictImg);
  //m_clPredictKern.setArg(12, cl::Local(sizeof(FeatType)*WG_WIDTH*WG_HEIGHT*FeatDim));
  m_clPredictKern.setArg(12, cl::Local(sizeof(FeatType)*256*FeatDim));

  // - per-image histogram update
  m_clPerImgHistKern.setArg(1, nChannels);
  m_clPerImgHistKern.setArg(8, FeatDim);
  m_clPerImgHistKern.setArg(9, m_clFeatLowBoundsBuff);
  m_clPerImgHistKern.setArg(10, m_clFeatUpBoundsBuff);
  m_clPerImgHistKern.setArg(11, params.nThresholds);
  m_clPerImgHistKern.setArg(12, params.thrLowBound);
  m_clPerImgHistKern.setArg(13, params.thrUpBound);
  m_clPerImgHistKern.setArg(15, tree.getID());
  m_clPerImgHistKern.setArg(18, m_clTreeLeftChildBuff);
  m_clPerImgHistKern.setArg(19, m_clTreeValuesBuff);
  m_clPerImgHistKern.setArg(20, cl::Local(sizeof(FeatType)*8));
  m_clPerImgHistKern.setArg(21, cl::Local(sizeof(FeatType)*WG_LHIST_UPDATE_WIDTH*FeatDim));

  // - node's best feature/threshold learning
  //m_clLearnBestFeatKern.setArg(0, m_clHistogramBuff);
  //m_clLearnBestFeatKern.setArg(2, params.nFeatures);
  //m_clLearnBestFeatKern.setArg(3, params.nThresholds);
  //m_clLearnBestFeatKern.setArg(5, perThreadFeatThrPairs);
  //m_clLearnBestFeatKern.setArg(6, m_clBestFeaturesBuff);
  //m_clLearnBestFeatKern.setArg(7, m_clBestThresholdsBuff);
  //m_clLearnBestFeatKern.setArg(8, m_clBestEntropiesBuff);


  // Init corresponding host buffers
  /** \todo use mapping/unmapping to avoid device/host copy */
  //m_tsNodesIDImg = new int[m_maxTsImgWidth*m_maxTsImgHeight*GLOBAL_HISTOGRAM_FIFO_SIZE];
  //m_perImgHist = new unsigned char[perImgHistogramSize*GLOBAL_HISTOGRAM_FIFO_SIZE];
  //m_bestFeatures = new unsigned int[learnBuffsSize];
  //m_bestThresholds = new unsigned int[learnBuffsSize];
  //m_bestEntropies = new float[learnBuffsSize];

  // Done with OpenCL initialization


  /*****************************************************************************************/
  /* NOTE: HARDCODED SIZE OF LAST HISTOGRAM DIMENSION, EQUAL TO 6, I.E. ACCUMULATOR FOR    */
  /* SINE AND COSINE OF THE SAMPLES EULER ANGLES                                           */
  /*****************************************************************************************/
  // Init the global histogram:
  // define the global histogram as a vector of per-node histograms. The total size of
  // the global histogram (defined as number of per-node histograms simultaneously kept)
  // is limited by the smaller between maxFrontierSize and
  // GLOBAL_HISTOGRAM_MAX_SIZE/perNodeHistogramSize
  // global histogram dimensions breakdown:
  // - node ID
  // - left/right
  // - feature
  // - threshold
  // - 12 // i.e. rotation matrix entries (rounded to closer 4 multiple)
  size_t maxFrontierSize = (endDepth>2) ? (2<<(endDepth-3)) : 1;
  size_t perNodeHistogramSize = 2*params.nFeatures*params.nThresholds*12;
  m_histogramSize = std::min(maxFrontierSize,
			     (size_t)floorl((double)GLOBAL_HISTOGRAM_MAX_SIZE/(perNodeHistogramSize*sizeof(double))));
  m_histogram = new double*[m_histogramSize];
  for (int i=0; i<m_histogramSize; i++) m_histogram[i] = new double[perNodeHistogramSize];


  // Buffer used to track to-train nodes for each depth
  m_frontier = new int[maxFrontierSize];


  // Init per-node total and per-class number of samples
  m_perNodeTotSamples = new unsigned int[nNodes];
  _initPerNodeCounters(tree, trainingSet, params, startDepth, endDepth);


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
	  unsigned int RDim>
unsigned int CLRTreeTrainer<ImgType, nChannels, FeatType, FeatDim, RDim>::_initFrontier(
  RTree<FeatType, FeatDim, RDim> &tree,
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
      const RTreeNode<FeatType, FeatDim, RDim> &currNode = tree.getNode(startNode+i);
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
	  unsigned int RDim>
unsigned int CLRTreeTrainer<ImgType, nChannels, FeatType, FeatDim, RDim>::_initHistogram(
  const TreeTrainerParameters<FeatType, FeatDim> &params)
{
  unsigned int frontierSize = m_frontierIdxMap.size();
  unsigned int nSlices = ceill((double)frontierSize/m_histogramSize);

  return nSlices;
}

template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int RDim>
void CLRTreeTrainer<ImgType, nChannels, FeatType, FeatDim, RDim>::_cleanTrain()
{
  // Release pinned memory objects
  m_clQueue1.enqueueUnmapMemObject(m_clTsImgPinn, m_clTsImgPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clTsMaskImgPinn, m_clTsMaskImgPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clTsNodesIDImgPinn, m_clTsNodesIDImgPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clTsSamplesBuffPinn, m_clTsSamplesBuffPinnPtr);
  m_clQueue1.enqueueUnmapMemObject(m_clPerImgHistBuffPinn, m_clPerImgHistBuffPinnPtr);


  // Delete data dinamically allocated for current tree training
  delete []m_perNodeTotSamples;
  delete []m_toSkipTsImg;
  delete []m_skippedTsImg;
  delete m_clTsImg1;
  delete m_clTsImg2;
  //delete []m_bestFeatures;
  //delete []m_bestThresholds;
  //delete []m_bestEntropies;
  for (int i=0; i<m_histogramSize; i++)
  {
    delete []m_histogram[i];
    m_histogram[i] = NULL;
  }
  delete []m_histogram;
  delete []m_frontier;
}


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int RDim>
void CLRTreeTrainer<ImgType, nChannels, FeatType, FeatDim, RDim>::_initPerNodeCounters(
  RTree<FeatType, FeatDim, RDim> &tree,
  const TrainingSetR<ImgType, nChannels, RDim> &trainingSet,
  const TreeTrainerParameters<FeatType, FeatDim> &params,
  unsigned int startDepth, unsigned int endDepth)
{
  unsigned int nNodes = (2<<(endDepth-1))-1;

  std::fill_n(m_perNodeTotSamples, nNodes, 0);

  /** \todo flat to-skip images here */
  const std::vector<TrainingSetRImage<ImgType, nChannels, RDim> > &tsImages =
    trainingSet.getImages();
  for (typename std::vector<TrainingSetRImage<ImgType, nChannels, RDim> >::const_iterator it = 
	 tsImages.begin();
       it!=tsImages.end(); ++it)
  {
    const TrainingSetRImage<ImgType, nChannels, RDim> &currImage=*it;
    
    if (startDepth==1)
    {
      m_perNodeTotSamples[0]+=currImage.getNSamples();
    }
    else
    {
      /** \todo A lot o redundant code from _impl_traverse. Move it within a function */
      /*********************************************************************************/
      /** \todo rewrite to support incremental training                                */
      /*********************************************************************************/

      /*
      int fillWidth, fillHeight;
      cl::size_t<3> origin, region;
      cl_int4 zeroColor = {0, 0, 0, 0};
      origin[0]=0; origin[1]=0, origin[2]=0;

      fillWidth = (currImage.getWidth()%WG_PREDICT_WIDTH) ? 
	WG_PREDICT_WIDTH-(currImage.getWidth()%WG_PREDICT_WIDTH) : 0;
      fillHeight = (currImage.getHeight()%WG_PREDICT_HEIGHT) ?
	WG_PREDICT_HEIGHT-(currImage.getHeight()%WG_PREDICT_HEIGHT) : 0;
      region[0]=m_maxTsImgWidth; region[1]=m_maxTsImgHeight; region[2]=1;

      m_clQueue1.enqueueFillImage(m_clTsNodesIDImg1, zeroColor, origin, region);

      // Note: if the current image is smaller than the previous one, part of the previous image
      // is accessible since not overwritten by current image
      // \todo zero filling of images
      region[0]=currImage.getWidth(); region[1]=currImage.getHeight();
      region[2] = (nChannels<=4) ? 1 : nChannels;
      std::copy(currImage.getData(), currImage.getData()+region[0]*region[1]*nChannels,
		m_clTsImgPinnPtr);
      if (nChannels<=4)
      {
	m_clQueue1.enqueueWriteImage(*reinterpret_cast<cl::Image2D*>(m_clTsImg1), CL_FALSE,
				     origin, region, 0, 0,
				     (void*)(m_clTsImgPinnPtr));
      }
      else
      {
	m_clQueue1.enqueueWriteImage(*reinterpret_cast<cl::Image3D*>(m_clTsImg1), CL_FALSE,
				     origin, region, 0, 0,
				     (void*)(m_clTsImgPinnPtr));
      }

      region[2] = 1;
      std::copy(currImage.getLabels(), currImage.getLabels()+region[0]*region[1],
		m_clTsLabelsImgPinnPtr);
      m_clQueue1.enqueueWriteImage(m_clTsLabelsImg1, CL_FALSE,
				   origin, region, 0, 0,
				   (void*)(m_clTsLabelsImgPinnPtr));
      std::copy(currImage.getSamples(), currImage.getSamples()+currImage.getNSamples(),
		m_clTsSamplesBuffPinnPtr);
      m_clQueue1.enqueueWriteBuffer(m_clTsSamplesBuff1, CL_FALSE,
				    0, currImage.getNSamples()*sizeof(cl_uint),
				    (void*)(m_clTsSamplesBuffPinnPtr));
    
      // Per-image prediction (i.e. compute samples end nodes)
      if (nChannels<=4)
      {
	m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image2D*>(m_clTsImg1));
      }
      else
      {
	m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image3D*>(m_clTsImg1));
      }
      m_clPredictKern.setArg(1, m_clTsLabelsImg1);
      m_clPredictKern.setArg(3, currImage.getWidth());
      m_clPredictKern.setArg(4, currImage.getHeight());
      m_clPredictKern.setArg(10, m_clTsNodesIDImg1);
      m_clPredictKern.setArg(11, m_clPredictImg1);
      
      for (unsigned int d=0; d<startDepth-1; d++)
      {
	m_clQueue1.enqueueNDRangeKernel(m_clPredictKern,
					cl::NullRange,
					cl::NDRange(currImage.getWidth()+fillWidth,
						    currImage.getHeight()+fillHeight),
					cl::NDRange(WG_PREDICT_WIDTH, WG_PREDICT_HEIGHT));
	m_clQueue1.enqueueCopyImage(m_clPredictImg1, m_clTsNodesIDImg1,
				    origin, origin, region);
      }

      m_clQueue1.enqueueReadImage(m_clTsNodesIDImg1, CL_TRUE,
				  origin, region, 0, 0,
				  (void*)(m_clTsNodesIDImgPinnPtr));

      for (unsigned int s=0; s<currImage.getNSamples();	 s++)
      {
	unsigned int id = currImage.getSamples()[s];
	unsigned int label = (unsigned int)currImage.getLabels()[id]-1;
	int nodeID = m_clTsNodesIDImgPinnPtr[id];

	m_perClassTotSamples[nodeID*nClasses+label]++;
	m_perNodeTotSamples[nodeID]++;
      }
      */
    }
  }
}
