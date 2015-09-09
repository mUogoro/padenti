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

__kernel void learnBestFeature(__global unsigned int *histogram,
			       __global unsigned int *perClassTotSamples,
			       unsigned int nFeatures, unsigned int nThresholds,
			       unsigned int nClasses,
			       unsigned int perThreadFeatThrPairs,
			       __global unsigned int *bestFeatures,
			       __global unsigned int *bestThresholds,
			       __global float *entropies
			       )
{
  float tmp, h, hL, hR, currBestEntropy, eps=1.e-12;
  int offset, nL, nR, currBestFeature, currBestThreshold;
  int nodeID, featureID, thrID;

  // Suppose the first feature and threshold as the best ones
  currBestFeature = get_global_id(0);
  currBestThreshold = 0;

  for (int i=0; i<perThreadFeatThrPairs; i++)
  {
    h = 0.0f;
    hL = 0.0f;
    hR = 0.0f;
    nL = 0;
    nR = 0;

    // Compute the current node, feature and threshold IDs and build the
    // corresponding offset within the histogram
    // Note: historam dimensions:
    // - node ID
    // - threshold ID
    // - feature ID
    offset = get_global_id(0)*perThreadFeatThrPairs+i;
    nodeID = offset/(nFeatures*nThresholds);
    //featureID = (offset%(nFeatures*nThresholds))/nThresholds;
    //thrID = offset%nThresholds;
    thrID = (offset%(nFeatures*nThresholds))/nFeatures;
    featureID = offset%nFeatures;

    // Compute the number of samples which, for the current feature/threshold pair,
    // reach the left and right branch respectively
    for (int l=0; l<nClasses; l++)
    {
      offset =\
	nodeID    * (nClasses*nThresholds*nFeatures) + 
	l         * (nThresholds*nFeatures) +
	thrID     * (nFeatures) +
	featureID;
      nL += histogram[offset];
      nR += perClassTotSamples[nodeID*nClasses+l]-histogram[offset];
    }

    // Compute the information gain and update the best one if improved
    for (int l=0; l<nClasses; l++)
    {
      offset =\
	nodeID    * (nClasses*nThresholds*nFeatures) + 
	l         * (nThresholds*nFeatures) +
	thrID     * (nFeatures) +
	featureID;

      //tmp = (float)(histogram[offset] + histogram[offset+1])/(nL+nR);
      tmp = ((float)perClassTotSamples[nodeID*nClasses+l])/(nL+nR);
      h -= (tmp<=eps) ? 0.0f : tmp*log2(tmp);

      tmp = (nL) ? (float)histogram[offset]/nL : 0.0f;
      hL -= (tmp<=eps) ? 0.0f : tmp*log2(tmp);

      tmp = (nR) ? (float)(perClassTotSamples[nodeID*nClasses+l]-histogram[offset])/nR : 0.0f;
      hR -= (tmp<=eps) ? 0.0f : tmp*log2(tmp);
    }
    tmp = h - ((float)nL/(nL+nR)*hL + (float)nR/(nL+nR)*hR); // Final information gain

    if (!i || tmp>currBestEntropy)
    {
      currBestEntropy=tmp;
      currBestFeature=featureID;
      currBestThreshold=thrID;
    }
  }

  /**
   * \todo  in order to reduce global memory write, save only the feature/threshold pair index and
   *        compute the corresponding feature and threshold IDs host side
   */
  bestFeatures[get_global_id(0)] = currBestFeature;
  bestThresholds[get_global_id(0)] = currBestThreshold;
  entropies[get_global_id(0)] = currBestEntropy;
}


/*
__kernel void learnBestFeature(__global unsigned int *histogram,
			       __global unsigned int *perClassTotSamples,
			       unsigned int nFeatures, unsigned int nThresholds,
			       unsigned int nClasses,
			       unsigned int perThreadFeatThrPairs,
			       __global unsigned int *bestFeatures,
			       __global unsigned int *bestThresholds,
			       __global float *entropies
			       )
{
  float tmp, h, hL, hR, currBestEntropy, eps=1.e-12;
  int offset, nL, nR, currBestFeature, currBestThreshold;
  int nodeID, featureID, thrID;

  // Suppose the first feature and threshold as the best ones
  currBestFeature = get_global_id(0);
  currBestThreshold = 0;

  for (int i=0; i<perThreadFeatThrPairs; i++)
  {
    h = 0.0f;
    hL = 0.0f;
    hR = 0.0f;
    nL = 0;
    nR = 0;

    // Compute the current node, feature and threshold IDs and build the
    // corresponding offset within the histogram
    // Note: historam dimensions:
    // - node ID
    // - feature ID
    // - threshold ID
    offset = get_global_id(0)*perThreadFeatThrPairs+i;
    nodeID = offset/(nFeatures*nThresholds);
    featureID = (offset%(nFeatures*nThresholds))/nThresholds;
    thrID = offset%nThresholds;

    // Compute the number of samples which, for the current feature/threshold pair,
    // reach the left and right branch respectively
    for (int l=0; l<nClasses; l++)
    {
      offset =\
	nodeID    * (nClasses*nFeatures*nThresholds) + 
	l         * (nFeatures*nThresholds) +
	featureID * (nThresholds) +
	thrID;
      nL += histogram[offset];
      nR += perClassTotSamples[nodeID*nClasses+l]-histogram[offset];
    }

    // Compute the information gain and update the best one if improved
    for (int l=0; l<nClasses; l++)
    {
      offset =\
	nodeID    * (nClasses*nFeatures*nThresholds) + 
	l         * (nFeatures*nThresholds) +
	featureID * (nThresholds) +
	thrID;

      //tmp = (float)(histogram[offset] + histogram[offset+1])/(nL+nR);
      tmp = ((float)perClassTotSamples[nodeID*nClasses+l])/(nL+nR);
      h -= (tmp<=eps) ? 0.0f : tmp*log2(tmp);

      tmp = (nL) ? (float)histogram[offset]/nL : 0.0f;
      hL -= (tmp<=eps) ? 0.0f : tmp*log2(tmp);

      tmp = (nR) ? (float)(perClassTotSamples[nodeID*nClasses+l]-histogram[offset])/nR : 0.0f;
      hR -= (tmp<=eps) ? 0.0f : tmp*log2(tmp);
    }
    tmp = h - ((float)nL/(nL+nR)*hL + (float)nR/(nL+nR)*hR); // Final information gain

    if (!i || tmp>currBestEntropy)
    {
      currBestEntropy=tmp;
      currBestFeature=featureID;
      currBestThreshold=thrID;
    }
  }

   // \todo  in order to reduce global memory write, save only the feature/threshold pair index and
   //        compute the corresponding feature and threshold IDs host side
  bestFeatures[get_global_id(0)] = currBestFeature;
  bestThresholds[get_global_id(0)] = currBestThreshold;
  entropies[get_global_id(0)] = currBestEntropy;
}
*/
