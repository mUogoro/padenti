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

#include "mex.h"
#include <algorithm>
#include <sstream>

#include <padenti/training_set_image.hpp>
#include <padenti/training_set.hpp>
#include <padenti/uniform_image_sampler.hpp>
#include <padenti/cl_tree_trainer.hpp>
#include "padenti_base.hpp"


// Redefine templates
// TODO: support for arbitrary template parameters
typedef TreeTrainerParameters<FeatType, FEAT_SIZE> TreeTrainerParametersT;
typedef TrainingSet<ImgType, N_CHANNELS> TrainingSetT;
typedef TrainingSetImage<ImgType, N_CHANNELS> TrainingSetImageT;
typedef UniformImageSampler<ImgType, N_CHANNELS> SamplerT;
typedef Tree<FeatType, FEAT_SIZE, N_CLASSES> TreeT;
typedef CLTreeTrainer<ImgType, N_CHANNELS, FeatType, FEAT_SIZE, N_CLASSES> TreeTrainerT;


/*! \todo add arguments check (e.g. types validation etc.) */
void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  TreeTrainerParametersT params;
  int nTrees, depth, nSamples;

  // Get number and depth of RF trees
  mxArray *nTreesField = mxGetField(prhs[0], 0, "nTrees");
  nTrees = static_cast<int>(mxGetScalar(nTreesField));
  mxArray *depthField = mxGetField(prhs[0], 0, "depth");
  depth = static_cast<int>(mxGetScalar(depthField));

  // Get number of sampled pixels per-image
  mxArray *nSamplesField = mxGetField(prhs[0], 0, "nSamples");
  nSamples = static_cast<int>(mxGetScalar(nSamplesField));
  
  // Get training parameters
  mxArray *nFeaturesField = mxGetField(prhs[0], 0, "nFeatures");
  params.nFeatures = static_cast<unsigned int>(mxGetScalar(nFeaturesField));
  mxArray *nThresholdsField = mxGetField(prhs[0], 0, "nThresholds");
  params.nThresholds = static_cast<unsigned int>(mxGetScalar(nThresholdsField));

  double *bounds;
  mxArray *featLowBoundsField = mxGetField(prhs[0], 0, "featLowBounds");
  bounds = mxGetPr(featLowBoundsField);
  std::copy(bounds, bounds+FEAT_SIZE, params.featLowBounds);
  
  mxArray *featUpBoundsField = mxGetField(prhs[0], 0, "featUpBounds");
  bounds = mxGetPr(featUpBoundsField);
  std::copy(bounds, bounds+FEAT_SIZE, params.featUpBounds);
  
  mxArray *thrLowBoundField = mxGetField(prhs[0], 0, "thrLowBound");
  params.thrLowBound = static_cast<FeatType>(mxGetScalar(thrLowBoundField));
  
  mxArray *thrUpBoundField = mxGetField(prhs[0], 0, "thrUpBound");
  params.thrUpBound = static_cast<FeatType>(mxGetScalar(thrUpBoundField));
  
  mxArray *perLeafSamplesThrField = mxGetField(prhs[0], 0, "perLeafSamplesThr");
  params.perLeafSamplesThr = static_cast<int>(mxGetScalar(perLeafSamplesThrField));
  
  // TODO: parameterize as well
  params.perLeafSamplesThr = 100;
  
  mexPrintf("Start training tree up to depth %d\n", depth);
  mexPrintf("Parameters:\n"
	    "features:              %d\n"
	    "thresholds:            %d\n"
	    "features lower bounds: [%f %f %f %f %f %f %f %f %f %f]\n"
	    "features upper bounds: [%f %f %f %f %f %f %f %f %f %f]\n"
        "thresholds bound:      [%f, %f]\n",
	    params.nFeatures,
	    params.nThresholds,
	    params.featLowBounds[0],
	    params.featLowBounds[1],
	    params.featLowBounds[2],
	    params.featLowBounds[3],
	    params.featLowBounds[4],
	    params.featLowBounds[5],
	    params.featLowBounds[6],
	    params.featLowBounds[7],
	    params.featLowBounds[8],
	    params.featLowBounds[9],
	    params.featUpBounds[0],
	    params.featUpBounds[1],
	    params.featUpBounds[2],
	    params.featUpBounds[3],
	    params.featUpBounds[4],
	    params.featUpBounds[5],
	    params.featUpBounds[6],
	    params.featUpBounds[7],
	    params.featUpBounds[8],
	    params.featUpBounds[9],
	    params.thrLowBound,
	    params.thrUpBound);

  // Get number of classes
  int nClasses = static_cast<int>(mxGetScalar(prhs[3]));
  
  // Check if the second and third arguments are cell arrays
  if (!mxIsCell(prhs[1]) || !mxIsCell(prhs[2]))
  {
    mexErrMsgTxt("Images and labels must be stored within a cell!!!\n");  
  }
  
  // Get number of images/labels (must be equal)
  mwSize nDimImgCell = mxGetNumberOfDimensions(prhs[1]);
  mwSize nDimLabelsCell = mxGetNumberOfDimensions(prhs[2]);
  const mwSize *dimImgCell = mxGetDimensions(prhs[1]);
  const mwSize *dimLabelsCell = mxGetDimensions(prhs[2]);
  if (dimImgCell[1]!=dimLabelsCell[1])
  {
    mexErrMsgTxt("Different number of images and labels!!!\n");
  }
  int nImages = dimImgCell[1];
  
  
  // Initialize the trainer
  // TODO: parameterize feature computation kernel path
  TreeTrainerT trainer("/home/daniele/ieiit/workspace/padenti/wrappers/matlab", false);
  
  // Start training
  for (int t=0; t<nTrees; t++)
  {
    
    // Init the training set and sampler
    TrainingSetT trainingSet(nClasses);
    SamplerT sampler(nSamples, t);
  
    // Cycle through images:
    for (int i=0; i<nImages; i++)
    {
      unsigned int *perImgSamples = new unsigned int[nSamples];
      unsigned int perImgNSamples;
   
      // - get the current image array
      mwIndex imgCell[2];
      imgCell[0]=0; imgCell[1]=i;
      mxArray *imgArray = 
	mxGetCell(prhs[1], mxCalcSingleSubscript(prhs[1], 2, imgCell));
    
      // - get the current labels array
      mwIndex labelsCell[2];
      labelsCell[0]=0; labelsCell[1]=i;
      mxArray *labelsArray = 
	mxGetCell(prhs[2], mxCalcSingleSubscript(prhs[2], 2, labelsCell));
    
      // - check the type of images and labels
      if (!mxIsUint32(imgArray))
      {
	mexPrintf("Image %d is not of type uint32, skip\n", i+1);
	continue;
      }
      if (!mxIsUint8(labelsArray))
      {
	mexPrintf("Image %d is not of type uint8, skip\n", i+1);
	continue;
      }
  
      // - get number of channels
      mwSize nDimImg = mxGetNumberOfDimensions(imgArray);
      mwSize nDimLabels = mxGetNumberOfDimensions(labelsArray);
      const mwSize *dimImg = mxGetDimensions(imgArray);
      const mwSize *dimLabels = mxGetDimensions(labelsArray);
      mwSize imgChannels = (nDimImg==3) ?
	(N_CHANNELS<=4 ? dimImg[0] : dimImg[2]) : 0;
    
      if (imgChannels!=N_CHANNELS)
      {
	mexPrintf("Unsupported number of image channels (must be %d), "
		  "skip %d-th pair\n", N_CHANNELS, i+1);
	continue;
      }
      if (nDimLabels>3)
      {
	mexPrintf("Labels must be single channel, skip %d-th pair!!!\n", i+1);
	continue;
      }
    
      // - get images and labels size
      mwSize imgWidth = (N_CHANNELS<=4) ? dimImg[2] : dimImg[1];
      mwSize imgHeight = (N_CHANNELS<=4) ? dimImg[1] : dimImg[0];
      mwSize lblWidth = (N_CHANNELS<=4) ? dimLabels[2] : dimLabels[1];
      mwSize lblHeight = (N_CHANNELS<=4) ? dimLabels[1] : dimLabels[0];
    
      if (imgWidth!=lblWidth || imgHeight!=lblHeight)
      {
	mexPrintf("Different %d-th image and %d-th labels size, skip pair\n",
		  i+1, i+1);
	continue;
      }
    
      // Get image and labels data
      ImgType *imgData =
	reinterpret_cast<ImgType*>(mxGetData(imgArray));
      unsigned char *lblData =
	reinterpret_cast<unsigned char*>(mxGetData(labelsArray));
    
      // Note: due to the different storage mode of Matlab (column major)
      // w.r.t. OpenCL (row major), flip width/height when loading
      // images into Padenti. This is equivalent to working on images
      // rotate 90 degrees ccw. Furthermore, this allows to avoid
      // image channels transpose
      int clWidth = imgHeight;
      int clHeight = imgWidth;
    
      // Apply uniform sampling
      perImgNSamples = sampler.sample(imgData, lblData,
				      clWidth, clHeight, perImgSamples);
      trainingSet << TrainingSetImageT(imgData, clWidth, clHeight,
				       lblData, nClasses,
				       perImgSamples, perImgNSamples);
      delete []perImgSamples;
    }
  
    mexPrintf("Training set loaded: per-class  sampled data probability\n");
    for (int i=0; i<nClasses; i++)
    {
      mexPrintf("%f ",trainingSet.getPriors()[i]);
    }
    mexPrintf("\n");

    // Done with training set loading


    // Train the current train
    TreeT tree(t, depth);
    
    //try
    //{
        trainer.train(tree, trainingSet, params, 1, depth);
    //}
    //catch (cl::Error err)
    //{
    //  std::cerr << err.what() << ": " << err.err() << std::endl;
    //}
    
    mexPrintf("Tree %d trained\n", t);
        
    // Done. Save the trained tree
    std::stringstream treeName;
    treeName << "tree" << t << ".xml";
    tree.save(treeName.str());
    mexPrintf("Tree %d dumped to disk as %s\n", t, treeName.str().c_str());
    
  }
  
  return;
}
