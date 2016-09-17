#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <padenti/image.hpp>
#include <padenti/cv_image_loader.hpp>
#include <padenti/rtree.hpp>
#include <padenti/cl_regressor.hpp>


static const int RDIM = 4;
static const int FEAT_SIZE = 2;
typedef Image<unsigned short, 1> DepthT;
typedef Image<unsigned char, 1> MaskT;
typedef CVImageLoader<unsigned short, 1> ImageLoaderT;
typedef RTree<short int, FEAT_SIZE, RDIM> RTreeT;
typedef RTreeNode<short int, FEAT_SIZE, RDIM> RTreeNodeT;
typedef RTree<short int, FEAT_SIZE, RDIM> RTreeT;
typedef Image<int, 1> PredictionT;
typedef CLRegressor<unsigned short, 1, short, FEAT_SIZE, RDIM> RegressorT;

#define USE_CPU        (false)
#define SAMPLED_PIXELS (1024)

int main(int argc, const char *argv[])
{
  if (argc<3)
  {
    std::cout << "Usage: " << argv[0] << "DEPTHS_FILE.png RTREE0.xml RTREE1.xml ... RTREEn.xml" << std::endl;
    return 1;
  }

  int nTrees = argc-2;
  RTreeT *trees = new RTreeT[nTrees];
  RegressorT regressor(".", USE_CPU);
  ImageLoaderT imageLoader;

  for (int t=0; t<nTrees; t++)
  {
    trees[t].load(argv[t+2], t);    
    regressor << trees[t];
  }

  DepthT depthmap = imageLoader.load(argv[1]);
  MaskT mask(depthmap.getWidth(), depthmap.getHeight());
  PredictionT prediction(depthmap.getWidth(), depthmap.getHeight());

  // TODO: explain masking
  cv::Mat cvDepth(depthmap.getHeight(), depthmap.getWidth(), CV_16U,
		          reinterpret_cast<unsigned char*>(depthmap.getData()));
  cv::Mat cvMask(cvDepth.rows, cvDepth.cols, CV_8U,
	        	   reinterpret_cast<unsigned char*>(mask.getData()));
  cvMask.setTo(0);
  cvMask.setTo(1, cvDepth>0);

  // TODO: how to handle multiple trees?
  regressor.predict(0, depthmap, prediction, mask);

  // Sample pixels
  // TODO: sort samples by weight and only then perform sampling
  unsigned int *nonNullPixelsBuff = new unsigned int[depthmap.getWidth()*depthmap.getHeight()];
  int nonNullPixels=0, nSamples=0;
  for (unsigned int id=0; id<depthmap.getWidth()*depthmap.getHeight(); id++)
  {
    if (mask.getData()[id]) nonNullPixelsBuff[nonNullPixels++]=id;
  }
  nSamples = std::min(SAMPLED_PIXELS,  nonNullPixels);

  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(0, nonNullPixels-1);
  unsigned int samples[SAMPLED_PIXELS];
  if (nSamples==SAMPLED_PIXELS)
  {
    for (unsigned int i=0; i<SAMPLED_PIXELS; i++)
    {
	  samples[i] = nonNullPixelsBuff[dist(gen)];
    }
  }
  else
  {
    std::copy(nonNullPixelsBuff, nonNullPixelsBuff+nSamples, samples);
  }

  // Compute rotation avg as in:
  // https://en.wikipedia.org/wiki/Mean_of_circular_quantities
  // Note: use only the first tree
  double sinSum[3] = {0, 0, 0};
  double cosSum[3] = {0, 0, 0};
  for (int i=0; i<nSamples; i++)
  {
    int nodeID = prediction.getData()[samples[i]];
    const RTreeNodeT &currNode = trees[0].getNode(nodeID);

    sinSum[0] += sin(currNode.m_value[0]);
    sinSum[1] += sin(currNode.m_value[1]);
    sinSum[2] += sin(currNode.m_value[2]);
    
	cosSum[0] += cos(currNode.m_value[0]);
    cosSum[1] += cos(currNode.m_value[1]);
    cosSum[2] += cos(currNode.m_value[2]);
  }

  double X = atan2(sinSum[2], cosSum[2])*180/M_PI;
  double Y = atan2(sinSum[1], cosSum[1])*180/M_PI;
  double Z = atan2(sinSum[0], cosSum[0])*180/M_PI;
  std::cout << "Rotation:" << std::endl
            << "X: " << X << " deg" << std::endl 
			<< "Y: " << Y << " deg" << std::endl
			<< "Z: " << Z << " deg" << std::endl;
  

  delete []nonNullPixelsBuff;
  delete []trees;

  return 0;
}
