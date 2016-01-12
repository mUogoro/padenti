#include <feat_type.cl>


#define WIN_WIDTH (32)
#define WIN_HEIGHT (16)

#define BG_BACKGROUND (10000)


feat_t computeFeature(__read_only image_t image,
		      uint nChannels, uint width, uint height, int2 coords,
		      __global int *treeLeftChildren,
		      __global float *treePosteriors,
		      __read_only image2d_t imageNodesID,
		      __local feat_t *features, uint featDim)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int r1, r2;
  int2 rCornerCoords;
  feat_t rParams0, rParams1;
  int channel;

  if ((coords.x+WIN_WIDTH)>width || (coords.y+WIN_HEIGHT)>height) return BG_BACKGROUND;
	
  // Read parameters (offsets and region size) for first region
  channel = ACCESS_FEATURE(features, 2, featDim);

  rParams0 = ACCESS_FEATURE(features, 0, featDim); rCornerCoords.x = coords.x + rParams0;
  rParams1 = ACCESS_FEATURE(features, 1, featDim); rCornerCoords.y = coords.y + rParams1;

  r1 = read_imageui(image, sampler, (int4)(rCornerCoords.x, rCornerCoords.y, channel, 0)).x;
  
  // Do the same for second block
  channel = ACCESS_FEATURE(features, 5, featDim);

  rParams0 = ACCESS_FEATURE(features, 3, featDim); rCornerCoords.x = coords.x + rParams0;
  rParams1 = ACCESS_FEATURE(features, 4, featDim); rCornerCoords.y = coords.y + rParams1;

  r2 = read_imageui(image, sampler, (int4)(rCornerCoords.x, rCornerCoords.y, channel, 0)).x;
  
  // TODO: delete
  //printf("[%ld X %ld] [%d X %d] %d %d %d %d %d %d, %d\n", 
  //	 get_global_size(0), get_global_size(1),
  //	 width, height,
  //	 coords.x, coords.y,
  //	 rCornerCoords.x, rCornerCoords.y,
  //	 rParams0, rParams1, (int)r2 );

  // Finally, return response
  return (feat_t)(r1-r2);
}
