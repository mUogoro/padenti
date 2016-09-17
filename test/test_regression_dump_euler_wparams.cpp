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

#include "rchord_meanshift.hpp"


#define RDIM (4)

// MeanShift number of hypothesis
#define MS_NHYPH    (2)

static const int FEAT_SIZE = 2;
typedef Image<unsigned short, 1> DepthT;
typedef Image<unsigned char, 1> MaskT;
typedef CVImageLoader<unsigned short, 1> ImageLoaderT;
typedef RTree<short int, FEAT_SIZE, RDIM> RTreeT;
typedef RTreeNode<short int, FEAT_SIZE, RDIM> RTreeNodeT;
typedef RTree<short int, FEAT_SIZE, RDIM> RTreeT;
typedef Image<int, 1> PredictionT;
typedef CLRegressor<unsigned short, 1, short, FEAT_SIZE, RDIM> RegressorT;

#define USE_CPU (false)

#define ROT_SCALE_X (1.f) //(0.95)
#define ROT_SCALE_Y (1.f) //(0.95)

int main(int argc, const char *argv[])
{
  if (argc<9)
  {
    std::cout << "Usage: " << argv[0]
	      << "DEPTHS_FILE.txt MS_BANDWIDTH MS_STEPSIZE MS_STEPTHR MS_GROUPINGTHR MS_NITERTHR MS_SAMPLEDPIXELS TREE0.xml ... TREEn.xml" << std::endl;
    return 1;
  }

  int nTrees = argc-8;
  RTreeT trees[nTrees];
  RegressorT regressor(".", USE_CPU);
  ImageLoaderT imageLoader;

  // Read meanshift parameters from command line
  float MS_SIGMA = atof(argv[2]);
  float MS_STEPSIZE = atof(argv[3]);
  float MS_STEPTHR = atof(argv[4]);
  float MS_SAMECTHR = atof(argv[5]);
  float MS_NITERTHR = atoi(argv[6]);
  int SAMPLED_PIXELS = atoi(argv[7]);

  for (int t=0; t<nTrees; t++)
  {
    trees[t].load(argv[t+8], t);

    if (ROT_SCALE_X || ROT_SCALE_Y)
    {
      size_t nNodes = (2<<(trees[t].getDepth()-1))-1;
      for (int n=0; n<nNodes; n++)
      {
	const RTreeNodeT &currTreeNode = trees[t].getNode(n);
	currTreeNode.m_feature[0] = 
	  static_cast<short>(static_cast<float>(currTreeNode.m_feature[0])*ROT_SCALE_X);
	currTreeNode.m_feature[1] = 
	  static_cast<short>(static_cast<float>(currTreeNode.m_feature[1])*ROT_SCALE_Y);
      }
    }
    
    regressor << trees[t];
  }

  std::ifstream dataPathsFile(argv[1]);
  std::ofstream eulFile("eul.txt");

  std::string path;
  while (getline(dataPathsFile, path))
  {
    DepthT depthmap = imageLoader.load(path);
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
    unsigned int nonNullPixelsBuff[depthmap.getWidth()*depthmap.getHeight()];
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

    // Compute rotation votes
    std::vector<Eigen::Vector3f> votes;
    std::vector<float> votesW;
    std::vector<Eigen::Vector3f> modes(MS_NHYPH);
    std::vector<float> modesW(MS_NHYPH);
    for (int i=0; i<nSamples; i++)
    {
      int nodeID = prediction.getData()[samples[i]];
      const RTreeNodeT &currNode = trees[0].getNode(nodeID);
      Eigen::Vector3f vote;

      vote.z() = currNode.m_value[0];
      vote.y() = currNode.m_value[1];
      vote.x() = currNode.m_value[2];
      votes.push_back(vote);
      votesW.push_back(1);
    }

    MultiGuessRChordMeanShift(votes, votesW, votes, modes, modesW,
			      MS_SIGMA, MS_STEPSIZE, MS_STEPTHR, MS_NITERTHR, MS_SAMECTHR);

    eulFile << (modes.at(0).z()*180.f/M_PI) << " "
	    << (modes.at(0).y()*180.f/M_PI) << " "
	    << (modes.at(0).x()*180.f/M_PI) << std::endl;
  }

  eulFile.close();

  return 0;
}
