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

#include <feat_type.cl>
#include <image_type.cl>

#ifdef ACCESS_FEATURE
#undef ACCESS_FEATURE
#endif

#define ACCESS_FEATURE(__ptr, __id, __size)               \
  (__ptr)[(__id)          * get_local_size(0)*get_local_size(1) + \
	  get_local_id(0) * get_local_size(1) + \
          get_local_id(1)]

#include <feature.cl>



__kernel void predict(__read_only image_t image, __read_only image2d_t mask,
		      uint nChannels, uint width, uint height,
		      __global int *treeLeftChildren,
		      __global feat_t *treeFeatures, unsigned int featDim,
		      __global feat_t *treeThresholds,
		      __global float *treePosteriors,
		      __read_only image2d_t imageNodesID,
		      __write_only image2d_t outNodesID,
		      __local feat_t *featuresBuff)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coords = (int2)(get_global_id(0), get_global_id(1));
  unsigned char maskValue = read_imageui(mask, sampler, coords).x;

  if (get_global_id(0) < width && get_global_id(1) < height)
  {

    if (maskValue)
    {
      int nodeID = read_imagei(imageNodesID, sampler, coords).x;
      int outNodeID = treeLeftChildren[nodeID];

      if (outNodeID == -1)
      {
	write_imagei(outNodesID, coords, (int4)(nodeID, 0, 0, 0));
      }
      else
      {
	__global feat_t *feature = treeFeatures+nodeID*featDim;
	feat_t thr = treeThresholds[nodeID];

	for (int i=0; i<featDim; i++)
	  ACCESS_FEATURE(featuresBuff, i, featDim) = feature[i];

	feat_t response = computeFeature(image, nChannels, width, height, coords,
					 treeLeftChildren,
					 treePosteriors,
					 imageNodesID,
					 featuresBuff, featDim);
	outNodeID += (response<=thr) ? 0 : 1;
	write_imagei(outNodesID, coords, (int4)(outNodeID, 0, 0, 0));
      }
    }
    //else
    //{
    //  /** \TODO: should it be written host side? Or should it be simply left unwritten? */
    //  write_imagei(outNodesID, coords, (int4)(-1, 0, 0, 0));
    //}
  }
}


__kernel void computePosterior(__read_only image2d_t nodesID,
			       uint width, uint height, uint nClasses,
			       __global float *treePosteriors,
			       __read_only image2d_t mask,
			       __global float *posterior)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  if (get_global_id(0) < width && get_global_id(1) < height)
  {
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    unsigned char maskValue = read_imageui(mask, sampler, coords).x;

    if (maskValue)
    {
      int nodeID = read_imagei(nodesID, sampler, coords.xy).x;
      __global float *nodePosteriors = treePosteriors+nodeID*nClasses;

      for (int l=0; l<nClasses; l++)
      {
	float posteriorValue = nodePosteriors[l];
	int offset = 
	  l        * width*height +
	  coords.y * width +
	  coords.x;
	posterior[offset] = posteriorValue;
      }
    }
  }
}
