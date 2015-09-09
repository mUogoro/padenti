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

// The feat_type.cl file is generated at run-time and store a "feat_t" typedef
// definition where the feature items value and response type is defined.
#include <feat_type.cl>


#define TARGET_DEPTH (500.0f)
#define BG_RESPONSE (10000.0f)


feat_t computeFeature(__read_only image2d_t image,
		      uint nChannels, uint width, uint height, int2 coords,
		      __global int *treeLeftChildren,
		      __global float *treePosteriors,
		      __read_only image2d_t imageNodesID,
		      __local feat_t *features, uint featDim)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  feat_t depth, response;
  int2 offset;
  bool outOfBorder;

  // Read the depth value at current pixel
  depth = (feat_t)read_imageui(image, sampler, coords).x;
  response = depth;

  // Compute the dispacement vector with respect to current coordinates
  // Note: to access feature values we always use the ACCESS_FEATURE macro, where the
  // first value is the features pointer, the second value is the item index and the
  // last value is the feature dimension
  offset.x = 
    coords.x+(int)round((float)ACCESS_FEATURE(features, 0, featDim)*TARGET_DEPTH/depth);
  offset.y =
    coords.y+(int)round((float)ACCESS_FEATURE(features, 1, featDim)*TARGET_DEPTH/depth);
  
  // Check if the displacement vector is beyond the image borders. In that case,
  // the depth value is set to BG_RESPONSE
  outOfBorder = offset.x<0 || offset.x>=width || offset.y<0 || offset.y>=height;
  depth = (outOfBorder) ? (feat_t)BG_RESPONSE : (feat_t)read_imageui(image, sampler, offset).x;

  // Finally, return the feature response, defined as the difference between the depth values
  // at the pixel coords and at the neighbour pixel
  response -= (depth) ? depth : (feat_t)BG_RESPONSE;

  return response;
}
