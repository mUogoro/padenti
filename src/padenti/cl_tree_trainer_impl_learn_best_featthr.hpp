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

#include <boost/chrono/chrono.hpp>
//#include <boost/log/trivial.hpp>

template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int nClasses>
void CLTreeTrainer<ImgType, nChannels, FeatType, FeatDim, nClasses>::_learnBestFeatThr(
  Tree<FeatType, FeatDim, nClasses> &tree,
  const TreeTrainerParameters<FeatType, FeatDim> &params,
  unsigned int currDepth, unsigned int currSlice)
{
  // Compute per-node best feature/threshold pair for current depth using ID3 algorithm
  /**
   * \todo different learning algorithm?
   * \todo as above, parallelize execution, e.g. parallel read/write and computation
   * \todo generic (non-md5) prng?
   * \todo terminate conditions?
  */

  boost::chrono::steady_clock::time_point startLearn = 
    boost::chrono::steady_clock::now();

  size_t perNodeHistogramSize = nClasses*params.nFeatures*params.nThresholds;
  unsigned int frontierSize = m_frontierIdxMap.size();
  unsigned int toTrainNodes = ((currSlice+1)*m_histogramSize > frontierSize) ?
    (frontierSize%m_histogramSize) : m_histogramSize;

  size_t learnBuffSize = 
    ((PARALLEL_LEARNT_NODES<frontierSize) ? PARALLEL_LEARNT_NODES : frontierSize) * 
    (params.nFeatures*params.nThresholds)/PER_THREAD_FEAT_THR_PAIRS;
  if (!currSlice)
  {
    std::fill_n(m_bestFeatures, learnBuffSize, 0);
    std::fill_n(m_bestThresholds, learnBuffSize, 0);
    std::fill_n(m_bestEntropies, learnBuffSize, -1.);
  }


  unsigned int nIters = toTrainNodes/PARALLEL_LEARNT_NODES + \
    ((toTrainNodes%PARALLEL_LEARNT_NODES) ? 1 : 0);
  for (unsigned int i=0; i<nIters; i++)
  {
    unsigned int currNNodes = (!i && toTrainNodes<=PARALLEL_LEARNT_NODES) ? toTrainNodes : 
      ((i==(nIters-1) && toTrainNodes%PARALLEL_LEARNT_NODES) ? 
       (toTrainNodes%PARALLEL_LEARNT_NODES) : PARALLEL_LEARNT_NODES);
    unsigned int nThreads =
      currNNodes*(params.nFeatures*params.nThresholds)/PER_THREAD_FEAT_THR_PAIRS;
    unsigned int frontierOffset = currSlice*m_histogramSize + i*PARALLEL_LEARNT_NODES;

    for (unsigned int n=0; n<currNNodes; n++)
    {
      unsigned int nodeID = m_frontier[frontierOffset+n];
      m_clQueue1.enqueueWriteBuffer(m_clHistogramBuff,
				   CL_FALSE,
				   n*perNodeHistogramSize*sizeof(cl_uint),
				   perNodeHistogramSize*sizeof(cl_uint),
				   (void*)m_histogram[i*PARALLEL_LEARNT_NODES+n]);

      // Upload per-node per-class total number of samples on GPU
      m_clQueue1.enqueueWriteBuffer(m_clPerClassTotSamplesBuff,
				   CL_FALSE,
				   n*nClasses*sizeof(cl_uint),
				   nClasses*sizeof(cl_uint),
				   (void*)(&m_perClassTotSamples[nodeID*nClasses]));
    }

    /** \todo Check if nThreads is a multiple of 32 */
    m_clQueue1.enqueueNDRangeKernel(m_clLearnBestFeatKern,
				    cl::NullRange,
				    cl::NDRange(nThreads),
				    cl::NDRange(32));

    m_clQueue1.enqueueReadBuffer(m_clBestFeaturesBuff,
				CL_FALSE,
				0, nThreads*sizeof(cl_uint), m_bestFeatures);
    m_clQueue1.enqueueReadBuffer(m_clBestThresholdsBuff,
				CL_FALSE,
				0, nThreads*sizeof(cl_uint), m_bestThresholds);
    m_clQueue1.enqueueReadBuffer(m_clBestEntropiesBuff,
				CL_TRUE,
				0, nThreads*sizeof(cl_float), m_bestEntropies);

    unsigned int perNodeThreads = nThreads/currNNodes;
    for (unsigned int n=0; n<currNNodes; n++)
    {
      unsigned int *perNodeBestFeatures = &m_bestFeatures[perNodeThreads*n];
      unsigned int *perNodeBestThresholds = &m_bestThresholds[perNodeThreads*n];
      float *perNodeBestEntropies = &m_bestEntropies[perNodeThreads*n];
      
      unsigned int nodeID = m_frontier[frontierOffset+n];
      unsigned int perNodeSliceOffset = i*PARALLEL_LEARNT_NODES+n;

      unsigned int bestID = std::distance(perNodeBestEntropies,
	std::max_element(perNodeBestEntropies, perNodeBestEntropies+perNodeThreads));

      // Update children histograms and left children
      /** \todo more elegant way? the two for above are awful */
      /** \todo move the node update to another function */
      const TreeNode<FeatType, FeatDim> &currNode = tree.getNode(nodeID);
      const TreeNode<FeatType, FeatDim> &leftChildNode = tree.getNode(nodeID*2+1);
      const TreeNode<FeatType, FeatDim> &rightChildNode = tree.getNode(nodeID*2+2);
      unsigned int *leftHistogram = new unsigned int[nClasses];
      unsigned int *rightHistogram = new unsigned int[nClasses];
      unsigned int lSum=0, rSum=0;
      for (unsigned int l=0; l<nClasses; l++)
      {
	unsigned int *currHistogram = m_histogram[perNodeSliceOffset];
	/*
	unsigned int offset = 
	  l                               * (params.nFeatures*params.nThresholds) +
	  perNodeBestFeatures[bestID]    * (params.nThresholds) +
	  perNodeBestThresholds[bestID];
	*/
	unsigned int offset = 
	  l                             * (params.nThresholds*params.nFeatures) +
	  perNodeBestThresholds[bestID] * (params.nFeatures) +
	  perNodeBestFeatures[bestID];
	leftHistogram[l] = currHistogram[offset];
	rightHistogram[l] = m_perClassTotSamples[nodeID*nClasses+l]-leftHistogram[l];
	lSum += leftHistogram[l];
	rSum += rightHistogram[l];
      }

      // If all the pixels follow the same path, the best-feature is not discriminative enough
      // and the current node is kept as a leaf node
      /** \todo insert additional splitting stop criteria here, e.g. information gain threshold */
      if (!lSum || !rSum) continue;

      // Update the total number of samples reaching child nodes
      m_perNodeTotSamples[nodeID*2+1] = lSum;
      m_perNodeTotSamples[nodeID*2+2] = rSum;
      for (unsigned int l=0; l<nClasses; l++)
      {
	m_perClassTotSamples[(nodeID*2+1)*nClasses+l]=leftHistogram[l];
	m_perClassTotSamples[(nodeID*2+2)*nClasses+l]=rightHistogram[l];
	assert((leftHistogram[l]+rightHistogram[l])==m_perClassTotSamples[nodeID*nClasses+l]);
      }
      assert((lSum+rSum)==m_perNodeTotSamples[nodeID]);

      float *leftPosteriors = new float[nClasses];
      float *rightPosteriors = new float[nClasses];
      for (unsigned int l=0; l<nClasses; l++)
      {
	leftPosteriors[l] = ((float)leftHistogram[l])/lSum;
	rightPosteriors[l] = ((float)rightHistogram[l])/rSum;
      }

      std::copy(leftPosteriors, leftPosteriors+nClasses, leftChildNode.m_posterior);
      std::copy(rightPosteriors, rightPosteriors+nClasses, rightChildNode.m_posterior);
      *leftChildNode.m_leftChild=-1;
      *rightChildNode.m_leftChild=-1;
	

      // Finally, recompute feature and threshold values for current node
      /**
       * \todo move integer-to-float conversion to prng, i.e. assume prngs work on floats
       * \todo compute features directly inside kernel during learning
       */
      /*
      unsigned int seed[4] = {tree.getID()^m_seed,
			      nodeID^m_seed,
			      perNodeBestFeatures[bestID]^m_seed,
			      m_seed};
      */
      unsigned int seed[4] = {tree.getID(),
			      nodeID,
			      perNodeBestFeatures[bestID],
			      0};
      unsigned int state[4];

      if (params.featLut.size())
      {
	int idx, lutEntrySize = FeatDim/params.nFeatLutSamples;
	for (int n=0; n<params.nFeatLutSamples; )
	{
	  md5Rand(seed, state);
	
	  idx = floor((float)state[0]/0xFFFFFFFF*params.featLut.size()/lutEntrySize)*lutEntrySize;
	  std::copy(params.featLut.begin()+idx,
		    params.featLut.begin()+idx+lutEntrySize,
		    currNode.m_feature+n*lutEntrySize);
	  if ((++n)>=params.nFeatLutSamples) break;

	  idx = floor((float)state[1]/0xFFFFFFFF*params.featLut.size()/lutEntrySize)*lutEntrySize;
	  std::copy(params.featLut.begin()+idx,
		    params.featLut.begin()+idx+lutEntrySize,
		    currNode.m_feature+n*lutEntrySize);
	  if ((++n)>=params.nFeatLutSamples) break;

	  idx = floor((float)state[2]/0xFFFFFFFF*params.featLut.size()/lutEntrySize)*lutEntrySize;
	  std::copy(params.featLut.begin()+idx,
		    params.featLut.begin()+idx+lutEntrySize,
		    currNode.m_feature+n*lutEntrySize);
	  if ((++n)>=params.nFeatLutSamples) break;

	  idx = floor((float)state[3]/0xFFFFFFFF*params.featLut.size()/lutEntrySize)*lutEntrySize;
	  std::copy(params.featLut.begin()+idx,
		    params.featLut.begin()+idx+lutEntrySize,
		    currNode.m_feature+n*lutEntrySize);
	  ++n;
	}
      }
      else
      {
	for (unsigned int j=0; j<FeatDim; j+=4)
	{
	  md5Rand(seed, state);
	 
	  currNode.m_feature[j] = params.featLowBounds[j] +
	    (FeatType)(((float)state[0])/(0xFFFFFFFF)*(params.featUpBounds[j]-
						       params.featLowBounds[j]));
	  if ((j+1)>=FeatDim) break;

	  currNode.m_feature[j+1] = params.featLowBounds[j+1]  +
	    (FeatType)(((float)state[1])/(0xFFFFFFFF)*(params.featUpBounds[j+1]-
						       params.featLowBounds[j+1]));
	  if ((j+2)>=FeatDim) break;
	  
	  currNode.m_feature[j+2] = params.featLowBounds[j+2] +
	    (FeatType)(((float)state[2])/(0xFFFFFFFF)*(params.featUpBounds[j+2]-
						       params.featLowBounds[j+2]));
	  if ((j+3)>=FeatDim) break;	  

	  currNode.m_feature[j+3] = params.featLowBounds[j+3] +
	    (FeatType)(((float)state[3])/(0xFFFFFFFF)*(params.featUpBounds[j+3]-
						       params.featLowBounds[j+3]));
	  
	  std::copy(state, state+4, seed);
	}
      }

      
      seed[0] = tree.getID();
      seed[1] = nodeID;
      seed[2] = perNodeBestFeatures[bestID];
      seed[3] = 1;
      for (unsigned int j=0; j<(perNodeBestThresholds[bestID]/4+1); j++)
      {
	md5Rand(seed, state);
	std::copy(state, state+4, seed);
      }

      if (params.thrLut.size())
      {
	int idx = floor((float)state[perNodeBestThresholds[bestID]%4]/0xFFFFFFFF *
			params.thrLut.size());
	*currNode.m_threshold = params.thrLut.at(idx);
      }
      else
      {
	*currNode.m_threshold = params.thrLowBound +
	  (FeatType)((float)state[perNodeBestThresholds[bestID]%4]/0xFFFFFFFF*
		     (params.thrUpBound-params.thrLowBound));
      }

      // Update current node left child
      *currNode.m_leftChild = nodeID*2+1;

      // Done with node

      delete []rightPosteriors;
      delete []leftPosteriors;
      delete []rightHistogram;
      delete []leftHistogram;
    }
  }

  // Finally, update node's portion of tree's OpenCL buffers
  //unsigned int startNode = m_frontier[currSlice*maxNodesPerGlobalHistogram];
  //unsigned int endNode = m_frontier[currSlice*maxNodesPerGlobalHistogram+toTrainNodes-1];
  unsigned int startNode = m_frontier[currSlice*m_histogramSize];
  unsigned int endNode = m_frontier[currSlice*m_histogramSize+toTrainNodes-1];
  unsigned int toWriteNodes = endNode-startNode+1;

  m_clQueue1.enqueueWriteBuffer(m_clTreeLeftChildBuff,
			       CL_FALSE,
			       startNode*sizeof(cl_int), toWriteNodes*sizeof(cl_int),
			       (void*)(&tree.getLeftChildren()[startNode]));
  m_clQueue1.enqueueWriteBuffer(m_clTreeLeftChildBuff,
			       CL_FALSE,
			       (startNode*2+1)*sizeof(cl_int), toWriteNodes*2*sizeof(cl_int),
			       (void*)(&tree.getLeftChildren()[startNode*2+1]));

  m_clQueue1.enqueueWriteBuffer(m_clTreeFeaturesBuff,
			       CL_FALSE,
			       startNode*FeatDim*sizeof(FeatType),
			       toWriteNodes*FeatDim*sizeof(FeatType),
			       (void*)(&tree.getFeatures()[startNode*FeatDim]));
  m_clQueue1.enqueueWriteBuffer(m_clTreeThrsBuff,
			       CL_FALSE,
			       startNode*sizeof(FeatType), toWriteNodes*sizeof(FeatType),
			       (void*)(&tree.getThresholds()[startNode]));
  m_clQueue1.enqueueWriteBuffer(m_clTreePosteriorsBuff,
			       CL_TRUE,
			       (startNode*2+1)*nClasses*sizeof(cl_float),
			       toWriteNodes*2*nClasses*sizeof(cl_float),
			       (void*)(&tree.getPosteriors()[(startNode+2+1)*nClasses]));

  /*
  boost::chrono::duration<double> learnTime = 
    boost::chrono::duration_cast<boost::chrono::duration<double> >(boost::chrono::steady_clock::now()-startLearn);
  BOOST_LOG_TRIVIAL(info) << "Best feature/threshold for nodes " << startNode
			  << "-" << endNode << " learnt in "
			  << learnTime.count() << " seconds";
  */
}
