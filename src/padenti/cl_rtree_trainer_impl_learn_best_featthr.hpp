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

#include <cfloat>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <boost/chrono/chrono.hpp>
//#include <boost/log/trivial.hpp>


template <typename ImgType, unsigned int nChannels, typename FeatType, unsigned int FeatDim,
	  unsigned int RDim>
void CLRTreeTrainer<ImgType, nChannels, FeatType, FeatDim, RDim>::_learnBestFeatThr(
  RTree<FeatType, FeatDim, RDim> &tree,
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

  //size_t perNodeHistogramSize = nClasses*params.nFeatures*params.nThresholds;
  size_t perNodeHistogramSize = 2*params.nFeatures*params.nThresholds*12;
  unsigned int frontierSize = m_frontierIdxMap.size();
  unsigned int toTrainNodes = ((currSlice+1)*m_histogramSize > frontierSize) ?
    (frontierSize%m_histogramSize) : m_histogramSize;


  // Naive serial gain-of-information computation
  unsigned int frontierOffset = currSlice*m_histogramSize;
  #pragma omp parallel for
  for (int i=0; i<toTrainNodes; i++)
  {
    unsigned int nodeID = m_frontier[frontierOffset+i];
    double *featThrScore = new double[params.nFeatures*params.nThresholds];
    double *gHistLPtr = m_histogram[i];
    double *gHistRPtr = m_histogram[i]+params.nThresholds*params.nFeatures*12;

    for (int t=0; t<params.nThresholds; t++)
    {
      for (int f=0; f<params.nFeatures; f++)
      {
	size_t offset = t*params.nFeatures*12 + f*12;
	double *ftHistLPtr = gHistLPtr+offset;
	double *ftHistRPtr = gHistRPtr+offset;

	// Get the total number of samples for left and right child
	double totL = ftHistLPtr[11];
	double totR = ftHistRPtr[11];
	assert((totL+totR)==m_perNodeTotSamples[nodeID]); // TODO delete

	
	// **************************************************************************
	// Sum of differences score
	// **************************************************************************
	// Compute variance for current node, left and right child
	
	// If all the samples reach either the left or right child node,
	// set the worst possible score and continue to the next threshold/feature pair
	if (!totL || !totR)
	{
	  featThrScore[f*params.nThresholds+t] = FLT_MAX;
	  continue;
	}

	// Compute the average rotation using SVD
	Eigen::MatrixXd sumL = Eigen::Matrix3d();
	sumL << ftHistLPtr[0], ftHistLPtr[4], ftHistLPtr[8],
	        ftHistLPtr[1], ftHistLPtr[5], ftHistLPtr[9],
	        ftHistLPtr[2], ftHistLPtr[6], ftHistLPtr[10];
	Eigen::MatrixXd sumR = Eigen::Matrix3d();
	sumR << ftHistRPtr[0], ftHistRPtr[4], ftHistRPtr[8],
	        ftHistRPtr[1], ftHistRPtr[5], ftHistRPtr[9],
                ftHistRPtr[2], ftHistRPtr[6], ftHistRPtr[10];
	Eigen::JacobiSVD<Eigen::Matrix3d> SVDL(sumL, Eigen::ComputeFullU|Eigen::ComputeFullV);
	Eigen::JacobiSVD<Eigen::Matrix3d> SVDR(sumR, Eigen::ComputeFullU|Eigen::ComputeFullV);
	
	Eigen::Matrix3d SL = SVDL.matrixU()*SVDL.matrixV().transpose();
	Eigen::Matrix3d SR = SVDR.matrixU()*SVDR.matrixV().transpose();
	if (SL.determinant()<0)
	{
	  SL = SVDL.matrixU()*Eigen::Vector3d(1,1,-1).asDiagonal()*SVDL.matrixV().transpose();
	}
	if (SR.determinant()<0)
	{
	  SR = SVDR.matrixU()*Eigen::Vector3d(1,1,-1).asDiagonal()*SVDR.matrixV().transpose();
	}	
	
	double scoreL = 6*totL-2*static_cast<float>(sumL.cwiseProduct(SL).sum());
	double scoreR = 6*totR-2*static_cast<float>(sumR.cwiseProduct(SR).sum());

	//featThrScore[f*params.nThresholds+t] =
	//  totL/(totL+totR)*(sumL.transpose()*SL).trace() + totR/(totL+totR)*(sumR.transpose()*SR).trace();

	featThrScore[f*params.nThresholds+t] = totL/(totL+totR)*scoreL + totR/(totL+totR)*scoreR;

	//featThrScore[f*params.nThresholds+t] =
	//  leftVar*totL/(totL+totR) + rightVar*totR/(totL+totR);
      }
    }

    // Find the best feature-threshold pair, i.e. the one that maximizes the gain of information
    //int bestFeatThrId = 
    //   static_cast<int>(std::max_element(featThrScore, featThrScore+params.nFeatures*params.nThresholds) -
    //			featThrScore);
    int bestFeatThrId =  
      static_cast<int>(std::min_element(featThrScore,
    					featThrScore + params.nFeatures*params.nThresholds) -
    		       featThrScore);
    int bestFeatId = bestFeatThrId/params.nThresholds;
    int bestThrId = bestFeatThrId%params.nThresholds;
    assert(featThrScore[bestFeatThrId]>=0);


    // **************** Same as original function from here on ********************
    // Update children histograms and left children
    /** \todo more elegant way? the two for above are awful */
    /** \todo move the node update to another function */
    const RTreeNode<FeatType, FeatDim, RDim> &currNode = tree.getNode(nodeID);
    const RTreeNode<FeatType, FeatDim, RDim> &leftChildNode = tree.getNode(nodeID*2+1);
    const RTreeNode<FeatType, FeatDim, RDim> &rightChildNode = tree.getNode(nodeID*2+2);
    unsigned int lSum=0, rSum=0;

    // Get the total number of samples for left and right child
    size_t offset = bestThrId*params.nFeatures*12 + bestFeatId*12;
    lSum = static_cast<unsigned int>(gHistLPtr[offset+11]);
    rSum = static_cast<unsigned int>(gHistRPtr[offset+11]);


    // If all the pixels follow the same path, the best-feature is not discriminative enough
    // and the current node is kept as a leaf node
    /** \todo insert additional splitting stop criteria here, e.g. information gain threshold */
    if (!lSum || !rSum) continue;

    // Update the total number of samples reaching child nodes
    m_perNodeTotSamples[nodeID*2+1] = lSum;
    m_perNodeTotSamples[nodeID*2+2] = rSum;
    assert((lSum+rSum)==m_perNodeTotSamples[nodeID]);

    // Update the node values
    // - Compute the average rotation and the corresponding Euler angles
    // TODO: avoid redundant code from loop above
    Eigen::MatrixXd sumL = Eigen::Matrix3d();
    sumL << gHistLPtr[offset+0], gHistLPtr[offset+4], gHistLPtr[offset+8],
            gHistLPtr[offset+1], gHistLPtr[offset+5], gHistLPtr[offset+9],
            gHistLPtr[offset+2], gHistLPtr[offset+6], gHistLPtr[offset+10];
    Eigen::MatrixXd sumR = Eigen::Matrix3d();
    sumR << gHistRPtr[offset+0], gHistRPtr[offset+4], gHistRPtr[offset+8],
            gHistRPtr[offset+1], gHistRPtr[offset+5], gHistRPtr[offset+9],
            gHistRPtr[offset+2], gHistRPtr[offset+6], gHistRPtr[offset+10];
    Eigen::JacobiSVD<Eigen::Matrix3d> SVDL(sumL, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::Matrix3d> SVDR(sumR, Eigen::ComputeFullU|Eigen::ComputeFullV);
    
    Eigen::Matrix3d SL = SVDL.matrixU()*SVDL.matrixV().transpose();
    Eigen::Matrix3d SR = SVDR.matrixU()*SVDR.matrixV().transpose();
    if (SL.determinant()<0)
    {
      SL = SVDL.matrixU()*Eigen::Vector3d(1,1,-1).asDiagonal()*SVDL.matrixV().transpose();
    }
    if (SR.determinant()<0)
    {
      SR = SVDR.matrixU()*Eigen::Vector3d(1,1,-1).asDiagonal()*SVDR.matrixV().transpose();
    }
    Eigen::Vector3d eul = SL.eulerAngles(2, 1, 0);
    leftChildNode.m_value[0] = static_cast<float>(eul.z());
    leftChildNode.m_value[1] = static_cast<float>(eul.y());
    leftChildNode.m_value[2] = static_cast<float>(eul.x());

    eul = SR.eulerAngles(2, 1, 0);
    rightChildNode.m_value[0] = static_cast<float>(eul.z());
    rightChildNode.m_value[1] = static_cast<float>(eul.y());
    rightChildNode.m_value[2] = static_cast<float>(eul.x());


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
			    bestFeatId,//perNodeBestFeatures[bestID],
			    0};
    unsigned int state[4];

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
  
    seed[0] = tree.getID();
    seed[1] = nodeID;
    seed[2] = bestFeatId;//perNodeBestFeatures[bestID];
    seed[3] = 1;
    //for (unsigned int j=0; j<(perNodeBestThresholds[bestID]/4+1); j++)
    for (unsigned int j=0; j<(bestThrId/4+1); j++)
    {
      md5Rand(seed, state);
      std::copy(state, state+4, seed);
    }

    *currNode.m_threshold = params.thrLowBound +
      //(FeatType)((float)state[perNodeBestThresholds[bestID]%4]/0xFFFFFFFF*
      (FeatType)((float)state[bestThrId%4]/0xFFFFFFFF*
		 (params.thrUpBound-params.thrLowBound));

    // Update current node left child
    *currNode.m_leftChild = nodeID*2+1;

    // Done with node
    delete []featThrScore;
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
  m_clQueue1.enqueueWriteBuffer(m_clTreeValuesBuff,
			       CL_TRUE,
			       (startNode*2+1)*RDim*sizeof(cl_float),
			       toWriteNodes*2*RDim*sizeof(cl_float),
			       (void*)(&tree.getValues()[(startNode+2+1)*RDim]));
}
