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
#include <padenti/cl_classifier.hpp>

// Note: so far we support only:
// - (integral) int images with 4 channels
// - float features of size 10
typedef unsigned int ImgType;
static const int N_CHANNELS = 4;
typedef float FeatType;
static const int FEAT_SIZE = 10;

typedef CLClassifier<ImgType, N_CHANNELS, FeatType, FEAT_SIZE, 21> ClassifierT;

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  // Check argument
  if (!mxIsUint64(prhs[0]))
  {
    mexErrMsgTxt("uint64 argument expected!!!\n");  
  }
    
  ClassifierT *classifierPtr = 
    reinterpret_cast<ClassifierT*>(*reinterpret_cast<uint64_T*>(mxGetData(prhs[0])));
  delete classifierPtr;
  
  return;
}