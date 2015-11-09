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
#include "padenti_base.hpp"

// Redefine templates
// TODO: support for arbitrary template parameters
typedef Tree<FeatType, FEAT_SIZE, N_CLASSES> TreeT;
typedef CLClassifier<ImgType, N_CHANNELS, FeatType, FEAT_SIZE, N_CLASSES> ClassifierT;


void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
   // Arguments check (must be all strings)
   for (int i=0; i<nrhs; i++)
   {
     if (!mxIsChar(prhs[i]))
     {
       mexErrMsgTxt("All arguments must be strings!!!\n");
     }
   }
   
   // Init the classifier
   // TODO: parameterize kernels path
   ClassifierT *classifier =
     new ClassifierT("/home/daniele/ieiit/workspace/padenti/wrappers/matlab", false);
   

   // Load Trees
   for (int t=0; t<nrhs; t++)
   {
     char treePath[1024];
     if (mxGetString(prhs[t], treePath, 1024))
     {
        mexErrMsgTxt("Unable to read all trees path!!!\n");
     }
     
     TreeT tree;
     tree.load(treePath, t);
     
     *classifier << tree;
   }
   
   // Return the classifier as unsigned long (i.e. return the classifier
   // pointer as its memory address). This value can be used to retrieve
   // the in-memory classifier instance by next calls
   mwSize outDim[2] = {1,1};
   plhs[0] = mxCreateNumericArray(2, outDim, mxUINT64_CLASS, mxREAL);
   uint64_T *classifierPtr = reinterpret_cast<uint64_T*>(mxGetData(plhs[0]));
   *classifierPtr = reinterpret_cast<uint64_T>(classifier);
   
   return;
}