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

#include <algorithm>
#include <queue>
#include <pthread.h>
#include <boost/chrono/chrono.hpp>
#include <boost/log/trivial.hpp>
#include <emmintrin.h>

template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
struct ConsumerProducerData
{
  int *nodesIDImg;
  unsigned int maxImgWidth;
  unsigned int maxImgHeight;
  unsigned int *perNodeTotSamples;
  unsigned int *perClassTotSamples;
  boost::unordered_map<int, int> *frontierIdxMap;
  unsigned int **histogram;
  unsigned char *perImgHistogram;
  size_t perImgHistogramStride;
  const TrainingSet<ImgType, nChannels> *trainingSet;
  bool *skippedTsImg;
  bool *toSkipTsImg;
  const TreeTrainerParameters<FeatType, FeatDim> *params;
  unsigned int currDepth;
  unsigned int frontierOffset;
  unsigned int startNode;
  unsigned int endNode;
  pthread_mutex_t *fifoMtx;
  pthread_cond_t *fifoCond;
  std::queue<int> *fifoQueue;
};
template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void *_updateGlobalHistogram(void *_data);


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::_traverseTrainingSet(
  const TrainingSet<ImgType, nChannels> &trainingSet,
  const TreeTrainerParameters<FeatType, FeatDim> &params,
  unsigned int currDepth, unsigned int currSlice)
{
  size_t perNodeHistogramSize = params.nFeatures*params.nThresholds*nClasses;
  size_t perImgHistogramStride = m_maxTsImgSamples*params.nFeatures*params.nThresholds;
  unsigned int frontierSize = m_frontierIdxMap.size();
  unsigned int frontierOffset = currSlice*m_histogramSize;

  unsigned int startNode = m_frontier[frontierOffset];
  unsigned int totNodes = ((frontierOffset+m_histogramSize)>frontierSize) ? \
    (frontierSize%m_histogramSize) : m_histogramSize;
  unsigned int endNode = m_frontier[frontierOffset+totNodes-1];

  // Fill global histogram with zeros
  #pragma omp parallel for
  for (int i=0; i<totNodes; i++) std::fill_n(m_histogram[i], perNodeHistogramSize, 0);

  // Consumer-producer stuff init
  std::queue<int> fifoQueue;
  pthread_mutex_t fifoMtx = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t fifoCond = PTHREAD_COND_INITIALIZER;
  struct ConsumerProducerData<ImgType, nChannels, FeatType, FeatDim, nClasses> consumerProducerData;
  consumerProducerData.nodesIDImg = m_clTsNodesIDImgPinnPtr;
  consumerProducerData.maxImgWidth = m_maxTsImgWidth;
  consumerProducerData.maxImgHeight = m_maxTsImgHeight;
  consumerProducerData.perNodeTotSamples = m_perNodeTotSamples;
  consumerProducerData.perClassTotSamples = m_perClassTotSamples;
  consumerProducerData.frontierIdxMap = &m_frontierIdxMap;
  consumerProducerData.histogram = m_histogram;
  consumerProducerData.perImgHistogram = m_clPerImgHistBuffPinnPtr;
  consumerProducerData.perImgHistogramStride = perImgHistogramStride;
  consumerProducerData.trainingSet = &trainingSet;
  consumerProducerData.skippedTsImg = m_skippedTsImg;
  consumerProducerData.toSkipTsImg = m_toSkipTsImg;
  consumerProducerData.params = &params;
  consumerProducerData.currDepth = currDepth;
  consumerProducerData.frontierOffset = frontierOffset;
  consumerProducerData.startNode = startNode;
  consumerProducerData.endNode = endNode;
  consumerProducerData.fifoMtx = &fifoMtx;
  consumerProducerData.fifoCond = &fifoCond;
  consumerProducerData.fifoQueue = &fifoQueue;


  // Start the consumer
  int queueIdx=0;
  pthread_t consumer;
  pthread_create(&consumer, NULL,
		 _updateGlobalHistogram<ImgType, nChannels, FeatType, FeatDim, nClasses>,
		 &consumerProducerData);


  // Start the producer
  
  // OpenCL events for timing purposes
  cl::Event startWriteEvent1, endWriteEvent1, startWriteEvent2, endWriteEvent2;
  cl::Event startComputeEvent1, endComputeEvent1, startComputeEvent2, endComputeEvent2;
  cl::Event startReadEvent, endReadEvent;
  cl_ulong totWriteTime=0, totComputeTime=0, totReadTime=0;
  
  int imgID=0, fillWidth, fillHeight;
  cl::size_t<3> origin, region;
  origin[0]=0; origin[1]=0, origin[2]=0;

  // Iterate through images and update the histogram
  const std::vector<TrainingSetImage<ImgType, nChannels> > &tsImages = trainingSet.getImages();
  for (typename std::vector<TrainingSetImage<ImgType, nChannels> >::const_iterator it=tsImages.begin();
       it!=tsImages.end(); ++it,++imgID)
  {
    const TrainingSetImage<ImgType, nChannels> &currImage = *it;

    cl::CommandQueue *weCLQueue = (imgID%2) ? &m_clQueue2 : &m_clQueue1;
    cl::CommandQueue *rCLQueue = (imgID%2) ? &m_clQueue1 : &m_clQueue2;
    size_t clPinnMemWOffset = (imgID%2) ? 1 : 0;


    if (!m_skippedTsImg[imgID])
    {
    // ************ FIRST QUEUE: WRITE AND KERNELS LAUNCH *************/
    fillWidth = (currImage.getWidth()%WG_PREDICT_WIDTH) ? 
      WG_PREDICT_WIDTH-(currImage.getWidth()%WG_PREDICT_WIDTH) : 0;
    fillHeight = (currImage.getHeight()%WG_PREDICT_HEIGHT) ?
      WG_PREDICT_HEIGHT-(currImage.getHeight()%WG_PREDICT_HEIGHT) : 0;
    region[0]=m_maxTsImgWidth; region[1]=m_maxTsImgHeight; region[2]=1;

    // TODO: check if its faster to perform bound-check on coordinates and erase only
    // the current image ROI instead of zeroing the whole image buffer 
    #ifdef CL_VERSION_1_2
      cl_int4 zeroColor = {0, 0, 0, 0};
      weCLQueue->enqueueFillImage((imgID%2) ? m_clTsNodesIDImg2 : m_clTsNodesIDImg1,
				  zeroColor, origin, region,
				  NULL, (imgID%2) ? &startWriteEvent2 : &startWriteEvent1);
    #else
      // TODO: find a way to use pinned memory or zero-ing kernel
      size_t rowPitch;
      char *tmpNodesIDptr = (char*)weCLQueue->enqueueMapImage((imgID%2) ? m_clTsNodesIDImg2 : m_clTsNodesIDImg1,
							      CL_TRUE, CL_MAP_WRITE,
							      origin, region, &rowPitch, NULL,
							      NULL, (imgID%2) ? &startWriteEvent2 :
							                        &startWriteEvent1);
      std::fill_n(tmpNodesIDptr, rowPitch*region[1], 0);
      weCLQueue->enqueueUnmapMemObject((imgID%2) ? m_clTsNodesIDImg2 : m_clTsNodesIDImg1,
				       tmpNodesIDptr);
    #endif


    // Note: if the current image is smaller than the previous one, part of the previous image
    // is accessible since not overwritten by current image
    /** \todo zero filling of images */
    region[0]=currImage.getWidth(); region[1]=currImage.getHeight();
    region[2] = (nChannels<=4) ? 1 : nChannels;
    std::copy(currImage.getData(), currImage.getData()+region[0]*region[1]*nChannels,
	      m_clTsImgPinnPtr+clPinnMemWOffset*region[0]*region[1]*nChannels);
    if (nChannels<=4)
    {
      weCLQueue->enqueueWriteImage(*reinterpret_cast<cl::Image2D*>((imgID%2) ?
								   m_clTsImg2 : m_clTsImg1),
				   CL_FALSE,
				   origin, region, 0, 0,
				   (void*)(m_clTsImgPinnPtr +
					   clPinnMemWOffset*region[0]*region[1]*nChannels));
    }
    else
    {
      weCLQueue->enqueueWriteImage(*reinterpret_cast<cl::Image3D*>((imgID%2) ?
								   m_clTsImg2 : m_clTsImg1),
				   CL_FALSE,
				   origin, region, 0, 0,
				   (void*)(m_clTsImgPinnPtr +
					   clPinnMemWOffset*region[0]*region[1]*nChannels));
    }

    region[2] = 1;
    std::copy(currImage.getLabels(), currImage.getLabels()+region[0]*region[1],
	      m_clTsLabelsImgPinnPtr+clPinnMemWOffset*region[0]*region[1]);
    weCLQueue->enqueueWriteImage((imgID%2) ? m_clTsLabelsImg2 : m_clTsLabelsImg1,
				 CL_FALSE,
				 origin, region, 0, 0,
				 (void*)(m_clTsLabelsImgPinnPtr + 
					 clPinnMemWOffset*region[0]*region[1]));
    std::copy(currImage.getSamples(), currImage.getSamples()+currImage.getNSamples(),
	      m_clTsSamplesBuffPinnPtr+clPinnMemWOffset*currImage.getNSamples());
    weCLQueue->enqueueWriteBuffer((imgID%2) ? m_clTsSamplesBuff2 : m_clTsSamplesBuff1,
				  CL_FALSE,
				  0, currImage.getNSamples()*sizeof(cl_uint),
				  (void*)(m_clTsSamplesBuffPinnPtr + 
					  clPinnMemWOffset*currImage.getNSamples()),
				  NULL, (imgID%2) ? &endWriteEvent2 : &endWriteEvent1);
    
    // Per-image prediction (i.e. compute samples end nodes)
    if (currDepth!=1)
    { 
      if (nChannels<=4)
      {
	m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image2D*>((imgID%2) ?
								  m_clTsImg2 : m_clTsImg1));
      }
      else
      {
	m_clPredictKern.setArg(0, *reinterpret_cast<cl::Image3D*>((imgID%2) ?
								  m_clTsImg2 : m_clTsImg1));
      }
      m_clPredictKern.setArg(1, (imgID%2) ? m_clTsLabelsImg2 : m_clTsLabelsImg1);
      m_clPredictKern.setArg(3, currImage.getWidth());
      m_clPredictKern.setArg(4, currImage.getHeight());
      m_clPredictKern.setArg(10, (imgID%2) ? m_clTsNodesIDImg2 : m_clTsNodesIDImg1);
      m_clPredictKern.setArg(11, (imgID%2) ? m_clPredictImg2 : m_clPredictImg1);

      for (unsigned int d=0; d<currDepth-1; d++)
      {
	weCLQueue->enqueueNDRangeKernel(m_clPredictKern,
					cl::NullRange,
					cl::NDRange(currImage.getWidth()+fillWidth,
						    currImage.getHeight()+fillHeight),
					//cl::NDRange(WG_WIDTH, WG_HEIGHT),
					cl::NDRange(WG_PREDICT_WIDTH, WG_PREDICT_HEIGHT),
					NULL,
					(d==0) ? ((imgID%2) ? &startComputeEvent2 :
                                                              &startComputeEvent1) : NULL);
	weCLQueue->enqueueCopyImage((imgID%2) ? m_clPredictImg2 : m_clPredictImg1,
				    (imgID%2) ? m_clTsNodesIDImg2 : m_clTsNodesIDImg1,
				    origin, origin, region);
      }
    }

 
    // Per-image histogram computation
    /** \todo Assure number of samples/#features are multiple of 8 */
    if (nChannels<=4)
    {
      m_clPerImgHistKern.setArg(0, *reinterpret_cast<cl::Image2D*>((imgID%2) ?
								   m_clTsImg2 : m_clTsImg1));
    }
    else
    {
      m_clPerImgHistKern.setArg(0, *reinterpret_cast<cl::Image3D*>((imgID%2) ?
								   m_clTsImg2 : m_clTsImg1));
    }
    m_clPerImgHistKern.setArg(2, currImage.getWidth());
    m_clPerImgHistKern.setArg(3, currImage.getHeight());
    m_clPerImgHistKern.setArg(4, (imgID%2) ? m_clTsLabelsImg2 : m_clTsLabelsImg1);
    m_clPerImgHistKern.setArg(5, (imgID%2) ? m_clTsNodesIDImg2 : m_clTsNodesIDImg1);
    m_clPerImgHistKern.setArg(6, (imgID%2) ? m_clTsSamplesBuff2 : m_clTsSamplesBuff1);
    m_clPerImgHistKern.setArg(7, currImage.getNSamples());
    m_clPerImgHistKern.setArg(14, (imgID%2) ? m_clPerImgHistBuff2 : m_clPerImgHistBuff1);
    m_clPerImgHistKern.setArg(16, startNode);
    m_clPerImgHistKern.setArg(17, endNode);
    weCLQueue->enqueueNDRangeKernel(m_clPerImgHistKern,
				    cl::NullRange,
				    cl::NDRange(currImage.getNSamples(), params.nFeatures),
				    //cl::NDRange(WG_WIDTH, WG_HEIGHT),
				    cl::NDRange(WG_LHIST_UPDATE_HEIGHT, WG_LHIST_UPDATE_WIDTH),
				    NULL, (imgID%2) ? &endComputeEvent2 : &endComputeEvent1);
    }

    // ************ SECOND QUEUE: READ PREVIOUS RESULTS *************/
    if (it!=tsImages.begin() && !m_skippedTsImg[imgID-1])
    {
      const TrainingSetImage<ImgType, nChannels> &prevImage = *(it-1);
      fillWidth = (prevImage.getWidth()%WG_PREDICT_WIDTH) ?
	WG_PREDICT_WIDTH-(prevImage.getWidth()%WG_PREDICT_WIDTH) : 0;
      fillHeight = (prevImage.getHeight()%WG_PREDICT_HEIGHT) ?
	WG_PREDICT_HEIGHT-(prevImage.getHeight()%WG_PREDICT_HEIGHT) : 0;
      region[0]=prevImage.getWidth(); region[1]=prevImage.getHeight();

      // Producer
      // Check if the queue is full
      pthread_mutex_lock(&fifoMtx);
      //if (fifoQueue.size()==GLOBAL_HISTOGRAM_FIFO_SIZE)
      while (fifoQueue.size()==GLOBAL_HISTOGRAM_FIFO_SIZE)
      {
	//std::cout << "P: queue full, wait ..."<< std::endl;
	pthread_cond_wait(&fifoCond, &fifoMtx);
      }
      pthread_mutex_unlock(&fifoMtx);


      if (currDepth==1 && it==tsImages.begin())
      {
	std::fill_n(m_clTsNodesIDImgPinnPtr,
		    GLOBAL_HISTOGRAM_FIFO_SIZE*m_maxTsImgWidth*m_maxTsImgHeight, 0);
      }
      else
      {
	rCLQueue->enqueueReadImage((imgID%2) ? m_clTsNodesIDImg1 : m_clTsNodesIDImg2,
				   CL_TRUE,
				   origin, region, 0, 0,
				   (void*)(m_clTsNodesIDImgPinnPtr+queueIdx*m_maxTsImgWidth*m_maxTsImgHeight),
				   NULL, &startReadEvent);
      }

      // Read per-image histogram
      rCLQueue->enqueueReadBuffer((imgID%2) ? m_clPerImgHistBuff1 : m_clPerImgHistBuff2,
				  CL_TRUE,
				  0,
				  prevImage.getNSamples()*params.nFeatures*params.nThresholds*sizeof(cl_uchar),
				  (void*)(m_clPerImgHistBuffPinnPtr+queueIdx*perImgHistogramStride),
				  NULL, &endReadEvent);
      
      // Queue the current per-image histogram and predicted end nodes
      pthread_mutex_lock(&fifoMtx);
      fifoQueue.push(queueIdx);
      //std::cout << "P: " << (queueIdx) << " produced" << std::endl;
      queueIdx++;
      queueIdx = queueIdx%GLOBAL_HISTOGRAM_FIFO_SIZE;
      pthread_mutex_unlock(&fifoMtx);
      pthread_cond_signal(&fifoCond);

      // Update timing info
      cl_ulong startTime, endTime;

      startTime = (imgID%2) ? startWriteEvent1.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
                              startWriteEvent2.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      endTime = (imgID%2) ? endWriteEvent1.getProfilingInfo<CL_PROFILING_COMMAND_END>() :
                            endWriteEvent2.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      totWriteTime += endTime-startTime;

      startTime = (currDepth!=1) ? 
	((imgID%2) ? startComputeEvent1.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
                     startComputeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_START>()):
	((imgID%2) ? endComputeEvent1.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
	             endComputeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_START>());
      endTime = (imgID%2) ? endComputeEvent1.getProfilingInfo<CL_PROFILING_COMMAND_END>() :
                            endComputeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      totComputeTime += endTime-startTime;

      startTime = (currDepth!=1) ?
	startReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
	endReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      endTime = endReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      totReadTime += endTime-startTime;
    }

    // Sync point: wait for computation to finish (reading is finished for sure since
    // local histogram read is blocking)
    if (!m_skippedTsImg[imgID])
      if (imgID%2) endComputeEvent2.wait(); else endComputeEvent1.wait();
  }

  //************ Last image **********
  --imgID;
  if (!m_skippedTsImg[imgID])
  {
    const TrainingSetImage<ImgType, nChannels> &currImage = *(tsImages.end()-1);
    cl::CommandQueue *rCLQueue = (imgID%2) ? &m_clQueue2 : &m_clQueue1;

    fillWidth = (currImage.getWidth()%WG_PREDICT_WIDTH) ?
      WG_PREDICT_WIDTH-(currImage.getWidth()%WG_PREDICT_WIDTH) : 0;
    fillHeight = (currImage.getHeight()%WG_PREDICT_HEIGHT) ?
      WG_PREDICT_HEIGHT-(currImage.getHeight()%WG_PREDICT_HEIGHT) : 0;
    region[0]=currImage.getWidth(); region[1]=currImage.getHeight();


    pthread_mutex_lock(&fifoMtx);
    while (fifoQueue.size()==GLOBAL_HISTOGRAM_FIFO_SIZE)
    {
      //std::cout << "P: queue full, wait ..."<< std::endl;
      pthread_cond_wait(&fifoCond, &fifoMtx);
    }
    pthread_mutex_unlock(&fifoMtx);
  
    if (currDepth!=1)
    {
      rCLQueue->enqueueReadImage((imgID%2) ? m_clTsNodesIDImg2 : m_clTsNodesIDImg1,
				 CL_FALSE,
				 origin, region, 0, 0,
				 (void*)(m_clTsNodesIDImgPinnPtr+queueIdx*m_maxTsImgWidth*m_maxTsImgHeight),
				 NULL, &startReadEvent);
    }

    // Read per-image histogram
    rCLQueue->enqueueReadBuffer((imgID%2) ? m_clPerImgHistBuff2 : m_clPerImgHistBuff1,
				CL_TRUE,
				0,
				currImage.getNSamples()*params.nFeatures*params.nThresholds*sizeof(cl_uchar),
				(void*)(m_clPerImgHistBuffPinnPtr+queueIdx*perImgHistogramStride),
				NULL, &endReadEvent);
  
    // Queue the current per-image histogram and predicted end nodes
    pthread_mutex_lock(&fifoMtx);
    fifoQueue.push(queueIdx);
    //std::cout << "P: " << (queueIdx) << " produced" << std::endl;
    queueIdx++;
    queueIdx = queueIdx%GLOBAL_HISTOGRAM_FIFO_SIZE;
    pthread_mutex_unlock(&fifoMtx);
    pthread_cond_signal(&fifoCond);
  
    cl_ulong startTime, endTime;

    startTime = (imgID%2) ? startWriteEvent2.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
                            startWriteEvent1.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    endTime = (imgID%2) ? endWriteEvent2.getProfilingInfo<CL_PROFILING_COMMAND_END>() :
                          endWriteEvent1.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    totWriteTime += endTime-startTime;

    startTime = (currDepth!=1) ? 
      ((imgID%2) ? startComputeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
                   startComputeEvent1.getProfilingInfo<CL_PROFILING_COMMAND_START>()):
      ((imgID%2) ? endComputeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
                   endComputeEvent1.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    endTime = (imgID%2) ? endComputeEvent2.getProfilingInfo<CL_PROFILING_COMMAND_END>() :
                          endComputeEvent1.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    totComputeTime += endTime-startTime;

    startTime = (currDepth!=1) ?
      startReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() :
      endReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    endTime = endReadEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    totReadTime += endTime-startTime;
  }


  // DONE with local histograms
  pthread_join(consumer, NULL);

  
  double totTime = static_cast<double>(totWriteTime)*1.e-9;
  BOOST_LOG_TRIVIAL(info) << "Total local histogram write time: "
			  << totTime
			  << " seconds";
                          // << (avg: " << totTime/tsImages.size()
			  // << " seconds)";
  totTime = static_cast<double>(totComputeTime)*1.e-9;
  BOOST_LOG_TRIVIAL(info) << "Total local histogram compute time: "
			  << totTime
			  << " seconds"; 
                          // << (avg: " << totTime/tsImages.size()
			  // << " seconds)";
  totTime = static_cast<double>(totReadTime)*1.e-9;
  BOOST_LOG_TRIVIAL(info) << "Total local histogram read time: "
			  << totTime
			  << " seconds";
                          // << (avg: " << totTime/tsImages.size()
			  //<< " seconds)";
  
}



template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void *_updateGlobalHistogram(void *_data)
{
  ConsumerProducerData<ImgType, nChannels, FeatType, FeatDim, nClasses> *data = 
    (ConsumerProducerData<ImgType, nChannels, FeatType, FeatDim, nClasses>*)_data;
  
  int *nodesIDImg = data->nodesIDImg;
  unsigned int maxImgWidth = data->maxImgWidth;
  unsigned int maxImgHeight = data->maxImgHeight;
  unsigned int *perNodeTotSamples = data->perNodeTotSamples;
  unsigned int *perClassTotSamples = data->perClassTotSamples;
  boost::unordered_map<int, int> *frontierIdxMap = data->frontierIdxMap;
  unsigned int **histogram = data->histogram;
  unsigned char *perImgHistogram = data->perImgHistogram;
  size_t perImgHistogramStride = data->perImgHistogramStride;
  const TrainingSet<ImgType, nChannels> &trainingSet = *data->trainingSet;
  bool *skippedTsImg = data->skippedTsImg;
  bool *toSkipTsImg = data->toSkipTsImg;
  const TreeTrainerParameters<FeatType, FeatDim> &params = *data->params;
  unsigned int currDepth = data->currDepth;
  unsigned int frontierOffset = data->frontierOffset;
  unsigned int startNode = data->startNode;
  unsigned int endNode = data->endNode;
  pthread_mutex_t &fifoMtx = *data->fifoMtx;
  pthread_cond_t &fifoCond = *data->fifoCond;
  std::queue<int> &fifoQueue = *data->fifoQueue;

  boost::chrono::duration<double> totGlobHistUpdateTime(0);

  int imgID = 0;
  const std::vector<TrainingSetImage<ImgType, nChannels> > &tsImages = trainingSet.getImages();
  for (typename std::vector<TrainingSetImage<ImgType, nChannels> >::const_iterator it=tsImages.begin();
       it!=tsImages.end(); ++it,++imgID)
  {
    if (skippedTsImg[imgID]) continue;

    const TrainingSetImage<ImgType, nChannels> &currImage = *it;
    int queueIdx;

    // Lock the queue and check if it's empty
    pthread_mutex_lock(&fifoMtx);
    while (fifoQueue.size()==0)
    {
      //std::cout << "C: queue empty ..." << std::endl;
      pthread_cond_wait(&fifoCond, &fifoMtx);
      // Get the id of the lastest unprocessed image histogram inside the queue
    }
    queueIdx = fifoQueue.front();
    pthread_mutex_unlock(&fifoMtx);

    boost::chrono::steady_clock::time_point startGlobHistUpdate = 
      boost::chrono::steady_clock::now();

    
    bool toSkipImg = true;
    size_t perImgOffset = queueIdx * perImgHistogramStride;
    size_t perImgNodeOffset = queueIdx*maxImgWidth*maxImgHeight;
    for (unsigned int s=0; s<currImage.getNSamples();
	 s++, perImgOffset+=(params.nThresholds*params.nFeatures))
    {
      unsigned int id = currImage.getSamples()[s];
      int nodeID = nodesIDImg[perImgNodeOffset+id];
      unsigned int label = (unsigned int)currImage.getLabels()[id]-1;

      // \todo move inside init 
      if (currDepth==1) perClassTotSamples[nodeID*nClasses+label]++;

      // If the current sample ends up in a node that belongs to a less deep level, skip it
      // \todo Sampe skipping criteria inside per-image histogram update kernel?
      if (nodeID<startNode || nodeID>endNode ||
	  perNodeTotSamples[nodeID]<=params.perLeafSamplesThr) continue;

      size_t globalOffset = label * (params.nFeatures*params.nThresholds);
      unsigned int *globalPtr = &histogram[frontierIdxMap->at(nodeID)-frontierOffset][globalOffset];
      unsigned char *localPtr = &perImgHistogram[perImgOffset];

      // SSE2 optimized version
      // \todo Check for SSE2 availability
      /*
      __m128i globalCounter;
      __m128i localCounter;
      for (unsigned int t=0; t<params.nThresholds; t++)
      {
	for (unsigned int f=0; f<params.nFeatures; f+=4, globalPtr+=4, localPtr+=4)
	{
	  globalCounter = _mm_loadu_si128(reinterpret_cast<__m128i*>(globalPtr));
	  
	  //localCounter = _mm_set_epi32(static_cast<int>(localPtr[3]),
	  //			       static_cast<int>(localPtr[2]),
	  //			       static_cast<int>(localPtr[1]),
	  //			       static_cast<int>(localPtr[0]));

	  localCounter = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr));
	  localCounter = _mm_unpacklo_epi8(localCounter, _mm_setzero_si128());
	  localCounter = _mm_unpacklo_epi16(localCounter, _mm_setzero_si128());

	  //localCounter = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr));
	  //localCounter = _mm_unpacklo_epi8(localCounter, localCounter);
	  //localCounter = _mm_unpacklo_epi16(localCounter, localCounter);
	  //localCounter = _mm_srai_epi32(localCounter, 24);

	  globalCounter = _mm_add_epi32(globalCounter, localCounter);
	  _mm_storeu_si128(reinterpret_cast<__m128i*>(globalPtr), globalCounter);
	  //_mm_stream_si128(reinterpret_cast<__m128i*>(globalPtr), globalCounter);
	}
      }
      */

      
      for (unsigned int t=0; t<params.nThresholds; t++)
      {
	for (unsigned int f=0; f<params.nFeatures; f+=16, globalPtr+=16, localPtr+=16)
	{
	  __m128i globalCounter1, globalCounter2, globalCounter3, globalCounter4;
	  __m128i localCounter1, localCounter2, localCounter3, localCounter4;
	  
	  globalCounter1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(globalPtr));
	  globalCounter2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(globalPtr+4));
	  globalCounter3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(globalPtr+8));
	  globalCounter4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(globalPtr+12));
	  
	  localCounter1 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr));
	  localCounter2 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr+4));
	  localCounter3 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr+8));
	  localCounter4 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr+12));


	  localCounter1 = _mm_unpacklo_epi8(localCounter1, _mm_setzero_si128());
	  localCounter1 = _mm_unpacklo_epi16(localCounter1, _mm_setzero_si128());

	  localCounter2 = _mm_unpacklo_epi8(localCounter2, _mm_setzero_si128());
	  localCounter2 = _mm_unpacklo_epi16(localCounter2, _mm_setzero_si128());

	  localCounter3 = _mm_unpacklo_epi8(localCounter3, _mm_setzero_si128());
	  localCounter3 = _mm_unpacklo_epi16(localCounter3, _mm_setzero_si128());

	  localCounter4 = _mm_unpacklo_epi8(localCounter4, _mm_setzero_si128());
	  localCounter4 = _mm_unpacklo_epi16(localCounter4, _mm_setzero_si128());


	  //localCounter = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr));
	  //localCounter = _mm_unpacklo_epi8(localCounter, localCounter);
	  //localCounter = _mm_unpacklo_epi16(localCounter, localCounter);
	  //localCounter = _mm_srai_epi32(localCounter, 24);

	  globalCounter1 = _mm_add_epi32(globalCounter1, localCounter1);
	  globalCounter2 = _mm_add_epi32(globalCounter2, localCounter2);
	  globalCounter3 = _mm_add_epi32(globalCounter3, localCounter3);
	  globalCounter4 = _mm_add_epi32(globalCounter4, localCounter4);

	  _mm_storeu_si128(reinterpret_cast<__m128i*>(globalPtr), globalCounter1);
	  _mm_storeu_si128(reinterpret_cast<__m128i*>(globalPtr+4), globalCounter2);
	  _mm_storeu_si128(reinterpret_cast<__m128i*>(globalPtr+8), globalCounter3);
	  _mm_storeu_si128(reinterpret_cast<__m128i*>(globalPtr+12), globalCounter4);
	  //_mm_stream_si128(reinterpret_cast<__m128i*>(globalPtr), globalCounter);
	}
      }
      

      /** \todo how to further improve throughput using OpenMP? */
      /*
      #pragma omp parallel for schedule(static,1) num_threads(4)
      for (unsigned int t=0; t<params.nThresholds; t++)
      {
	//unsigned int *pvtGlobalPtr = globalPtr + t*params.nFeatures; 
	//unsigned char *pvtLocalPtr = localPtr + t*params.nFeatures;

	//for (unsigned int f=0; f<params.nFeatures; f+=16, pvtGlobalPtr+=16, pvtLocalPtr+=16)
	for (unsigned int f=0; f<params.nFeatures; f+=16)
	{
	  __m128i globalCounter1, globalCounter2, globalCounter3, globalCounter4;
	  __m128i localCounter1, localCounter2, localCounter3, localCounter4;
	  
	  unsigned int *pvtGlobalPtr = globalPtr + t*params.nFeatures + f; 
	  unsigned char *pvtLocalPtr = localPtr + t*params.nFeatures + f;

	  globalCounter1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr));
	  globalCounter2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr+4));
	  globalCounter3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr+8));
	  globalCounter4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr+12));
	  
	  localCounter1 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(pvtLocalPtr));
	  localCounter2 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(pvtLocalPtr+4));
	  localCounter3 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(pvtLocalPtr+8));
	  localCounter4 = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(pvtLocalPtr+12));


	  localCounter1 = _mm_unpacklo_epi8(localCounter1, _mm_setzero_si128());
	  localCounter1 = _mm_unpacklo_epi16(localCounter1, _mm_setzero_si128());

	  localCounter2 = _mm_unpacklo_epi8(localCounter2, _mm_setzero_si128());
	  localCounter2 = _mm_unpacklo_epi16(localCounter2, _mm_setzero_si128());

	  localCounter3 = _mm_unpacklo_epi8(localCounter3, _mm_setzero_si128());
	  localCounter3 = _mm_unpacklo_epi16(localCounter3, _mm_setzero_si128());

	  localCounter4 = _mm_unpacklo_epi8(localCounter4, _mm_setzero_si128());
	  localCounter4 = _mm_unpacklo_epi16(localCounter4, _mm_setzero_si128());


	  //localCounter = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(localPtr));
	  //localCounter = _mm_unpacklo_epi8(localCounter, localCounter);
	  //localCounter = _mm_unpacklo_epi16(localCounter, localCounter);
	  //localCounter = _mm_srai_epi32(localCounter, 24);

	  globalCounter1 = _mm_add_epi32(globalCounter1, localCounter1);
	  globalCounter2 = _mm_add_epi32(globalCounter2, localCounter2);
	  globalCounter3 = _mm_add_epi32(globalCounter3, localCounter3);
	  globalCounter4 = _mm_add_epi32(globalCounter4, localCounter4);

	  _mm_storeu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr), globalCounter1);
	  _mm_storeu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr+4), globalCounter2);
	  _mm_storeu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr+8), globalCounter3);
	  _mm_storeu_si128(reinterpret_cast<__m128i*>(pvtGlobalPtr+12), globalCounter4);
	  //_mm_stream_si128(reinterpret_cast<__m128i*>(globalPtr), globalCounter);
	}
      }
      */
      
      toSkipImg = false;
    }
    
    if (!toSkipImg) toSkipTsImg[imgID] = false;
    
    totGlobHistUpdateTime += boost::chrono::steady_clock::now() - startGlobHistUpdate;

    // Dequeue the image histogram id and signal
    pthread_mutex_lock(&fifoMtx);
    //std::cout << "C: " << fifoQueue.front() << " consumed" << std::endl;
    fifoQueue.pop();
    pthread_mutex_unlock(&fifoMtx);
    pthread_cond_signal(&fifoCond);
  }
  

  boost::chrono::duration<double> totGlobHistUpdateSeconds = 
    boost::chrono::duration_cast<boost::chrono::duration<double> >(totGlobHistUpdateTime);
  
  BOOST_LOG_TRIVIAL(info) << "Total global histogram update time: " << totGlobHistUpdateSeconds.count()
                          //<< " seconds (avg: " << totGlobHistUpdateSeconds.count()/tsImages.size()
			  << " seconds)";
  
}
