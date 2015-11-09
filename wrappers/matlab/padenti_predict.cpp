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
#include <padenti/cl_classifier.hpp>
#include "padenti_base.hpp"


// Redefine templates
// TODO: support for arbitrary template parameters
typedef Image<ImgType, N_CHANNELS> ImageT;
typedef Tree<FeatType, FEAT_SIZE, N_CLASSES> TreeT;
typedef CLClassifier<ImgType, N_CHANNELS, FeatType, FEAT_SIZE, N_CLASSES> ClassifierT;
typedef Image<float, N_CLASSES> PredictionT;

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  // Check arguments type
  if (!mxIsUint64(prhs[0]))
  {
    mexErrMsgTxt("uint64 argument expected as first argument!!!\n");  
  }
  if (!mxIsUint32(prhs[1]))
  {
    mexErrMsgTxt("uint32 array expected as second argument!!!\n");  
  }
  
  // Check input image format
  mwSize nDimImage = mxGetNumberOfDimensions(prhs[1]);
  const mwSize *dimImage = mxGetDimensions(prhs[1]);
  mwSize imgChannels = (nDimImage==3) ? 
    (N_CHANNELS<=4 ? dimImage[0] : dimImage[2]) : 0;
  if (imgChannels!=N_CHANNELS)
  {
    mexErrMsgTxt("Input image must have at least 4 channels!!!\n");  
  }
  
  // Create the Padenti images used for storing the input image and the
  // prediction result
  // Note: due to the different storage mode of Matlab (column major)
  // w.r.t. OpenCL (row major), flip width/height when loading
  // images into Padenti. This is equivalent to working on images
  // rotate 90 degrees ccw. Furthermore, this allows to avoid
  // image channels transpose
  int imgWidth = (N_CHANNELS<=4) ? dimImage[2] : dimImage[1];
  int imgHeight = (N_CHANNELS<=4) ? dimImage[1] : dimImage[0];
  int clWidth = imgHeight;
  int clHeight = imgWidth;
  ImageT image(reinterpret_cast<ImgType*>(mxGetData(prhs[1])),
               clWidth, clHeight);
  PredictionT prediction(clWidth, clHeight);
  
  // Get the classifier instance
  ClassifierT *classifierPtr = 
    reinterpret_cast<ClassifierT*>(*reinterpret_cast<uint64_T*>(mxGetData(prhs[0])));
  
  // Perform prediction
  classifierPtr->predict(image, prediction);
  
  // Save prediction result into Matlab array
  mwSize outDim[3];
  outDim[0] = imgHeight;
  outDim[1] = imgWidth;
  outDim[2] = N_CLASSES;
  plhs[0] = mxCreateNumericArray(3, outDim, mxSINGLE_CLASS, mxREAL);
  
  float *outPixels = reinterpret_cast<float*>(mxGetData(plhs[0]));
  std::copy(prediction.getData(),
            prediction.getData()+clWidth*clHeight*N_CLASSES,
            outPixels);
  
  // Done
  return;
}