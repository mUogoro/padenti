#include <feat_type.cl>

#define BG_RESPONSE (10000)


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
  //uint4 intImgValue;
  //int *intImgValuePtr;
  int channel;
	
  // Read parameters (offsets and region size) for first region
  //rParams0= ACCESS_FEATURE(features, 4, featDim);
  //intImgValuePtr = &((int*)&intImgValue)[rParams0];
  channel = ACCESS_FEATURE(features, 4, featDim);

  rParams0 = ACCESS_FEATURE(features, 0, featDim); rCornerCoords.x = coords.x + rParams0;
  rParams1 = ACCESS_FEATURE(features, 1, featDim); rCornerCoords.y = coords.y + rParams1;

  rParams0 = ACCESS_FEATURE(features, 2, featDim);
  rParams1 = ACCESS_FEATURE(features, 3, featDim);

  if ((rCornerCoords.x + rParams0) >= width ||
      (rCornerCoords.x - rParams0) <  0     ||
      (rCornerCoords.y + rParams1) >= height||
      (rCornerCoords.y - rParams1) <  0)
  {
    return BG_RESPONSE;
  }
  else
  {
    // A corner
    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x-rParams0,
    //						      rCornerCoords.y-rParams1));
    //r1 = *intImgValuePtr;
    r1 = read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
					     rCornerCoords.y-rParams1,
					     channel, 0)).x;

    // B Corner
    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x+rParams0,
    //						      rCornerCoords.y-rParams1));
    //r1 -= *intImgValuePtr;
    r1 -= read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
					      rCornerCoords.y-rParams1,
					      channel, 0)).x;
		
    // C Corner
    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x-rParams0,
    //                                                  rCornerCoords.y+rParams1));
    //r1 -= *intImgValuePtr;
    r1 -= read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
					      rCornerCoords.y+rParams1,
					      channel, 0)).x;

    // D Corner
    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x+rParams0,
    //						      rCornerCoords.y+rParams1));
    //r1 += *intImgValuePtr;
    r1 += read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
					      rCornerCoords.y+rParams1,
					      channel, 0)).x;

    r1 = (feat_t)round((float)r1/((rParams0*2+1)*(rParams1*2+1)));
  }

  // Do the same for second block
  //rParams0 = ACCESS_FEATURE(features, 9, featDim);
  //intImgValuePtr = &((int*)&intImgValue)[rParams0];
  channel = ACCESS_FEATURE(features, 9, featDim);

  rParams0 = ACCESS_FEATURE(features, 5, featDim); rCornerCoords.x = coords.x + rParams0;
  rParams1 = ACCESS_FEATURE(features, 6, featDim); rCornerCoords.y = coords.y + rParams1;

  rParams0 = ACCESS_FEATURE(features, 7, featDim);
  rParams1 = ACCESS_FEATURE(features, 8, featDim);

  if ((rCornerCoords.x + rParams0) >= width ||
      (rCornerCoords.x - rParams0) <  0     ||
      (rCornerCoords.y + rParams1) >= height||
      (rCornerCoords.y - rParams1) <  0)
  {
    return BG_RESPONSE;
  }
  else
  {
    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x-rParams0,
    //						      rCornerCoords.y-rParams1));
    //r2 = *intImgValuePtr;
    r2 = read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
					     rCornerCoords.y-rParams1,
					     channel, 0)).x;

    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x+rParams0,
    //						      rCornerCoords.y-rParams1));
    //r2 -= *intImgValuePtr;
    r2 -= read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
					      rCornerCoords.y-rParams1,
					      channel, 0)).x;

    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x-rParams0,
    //						      rCornerCoords.y+rParams1));
    //r2 -= *intImgValuePtr;
    r2 -= read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
					      rCornerCoords.y+rParams1,
					      channel, 0)).x;
    
    //intImgValue = read_imageui(image, sampler, (int2)(rCornerCoords.x+rParams0,
    //						      rCornerCoords.y+rParams1));
    //r2 += *intImgValuePtr;
    r2 += read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
					      rCornerCoords.y+rParams1,
					      channel, 0)).x;

    r2 = (feat_t)round((float)r2/((rParams0*2+1)*(rParams1*2+1)));
  }

  // Finally, return response
  return (feat_t)(r1-r2);
}
