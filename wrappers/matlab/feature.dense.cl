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

  if (coords.x>=width || coords.y>=height) return BG_BACKGROUND;
	
  // Read parameters (offsets and region size) for first region
  channel = ACCESS_FEATURE(features, 4, featDim);

  rParams0 = ACCESS_FEATURE(features, 0, featDim); rCornerCoords.x = coords.x + rParams0;
  rParams1 = ACCESS_FEATURE(features, 1, featDim); rCornerCoords.y = coords.y + rParams1;

  rParams0 = ACCESS_FEATURE(features, 2, featDim);
  rParams1 = ACCESS_FEATURE(features, 3, featDim);

  //rParams0 = ((rCornerCoords.x+rParams0)>=(coords.x+WIN_WIDTH)) ? 
  //           coords.x+WIN_WIDTH-rCornerCoords.x-1 : rParams0;
  //rParams1 = ((rCornerCoords.y+rParams1)>=(coords.y+WIN_HEIGHT)) ? 
  //           coords.y+WIN_HEIGHT-rCornerCoords.y-1 : rParams1;

  // Check if the feature access outside path boundaries
  // Note: recicle r1 variables
  r1 = floor((float)coords.x/WIN_WIDTH)*WIN_WIDTH;
  if ((rCornerCoords.x-rParams0)<r1 ||
      (rCornerCoords.x+rParams0)>=(r1+WIN_WIDTH)) return BG_BACKGROUND;
  r1 = floor((float)coords.y/WIN_HEIGHT)*WIN_HEIGHT;
  if ((rCornerCoords.y-rParams1)<r1 ||
      (rCornerCoords.y+rParams1)>=(r1+WIN_HEIGHT)) return BG_BACKGROUND;

  // A corner
  r1 = read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
                                           rCornerCoords.y-rParams1,
                                           channel, 0)).x;
  
  // B Corner
  r1 -= read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
  					      rCornerCoords.y-rParams1,
  					      channel, 0)).x;
  		
  // C Corner
  r1 -= read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
					    rCornerCoords.y+rParams1,
					    channel, 0)).x;
  
  // D Corner
  r1 += read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
					    rCornerCoords.y+rParams1,
					    channel, 0)).x;
  
  r1 = (feat_t)round((float)r1/((rParams0*2+1)*(rParams1*2+1)));
  

  // Do the same for second block
  //rParams0 = ACCESS_FEATURE(features, 9, featDim);
  //intImgValuePtr = &((int*)&intImgValue)[rParams0];
  channel = ACCESS_FEATURE(features, 9, featDim);

  rParams0 = ACCESS_FEATURE(features, 5, featDim); rCornerCoords.x = coords.x + rParams0;
  rParams1 = ACCESS_FEATURE(features, 6, featDim); rCornerCoords.y = coords.y + rParams1;

  rParams0 = ACCESS_FEATURE(features, 7, featDim);
  rParams1 = ACCESS_FEATURE(features, 8, featDim);

  //rParams0 = ((rCornerCoords.x+rParams0)>=(coords.x+WIN_WIDTH)) ? 
  //           coords.x+WIN_WIDTH-rCornerCoords.x-1 : rParams0;
  //rParams1 = ((rCornerCoords.y+rParams1)>=(coords.y+WIN_HEIGHT)) ? 
  //           coords.y+WIN_HEIGHT-rCornerCoords.y-1 : rParams1;

  // Note: recicle r2 variables
  r2 = floor((float)coords.x/WIN_WIDTH)*WIN_WIDTH;
  if ((rCornerCoords.x-rParams0)<r2 ||
      (rCornerCoords.x+rParams0)>=(r2+WIN_WIDTH)) return BG_BACKGROUND;
  r2 = floor((float)coords.y/WIN_HEIGHT)*WIN_HEIGHT;
  if ((rCornerCoords.y-rParams1)<r2 ||
      (rCornerCoords.y+rParams1)>=(r2+WIN_HEIGHT)) return BG_BACKGROUND;

  
  r2 = read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
					   rCornerCoords.y-rParams1,
					   channel, 0)).x;
  
  r2 -= read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
					    rCornerCoords.y-rParams1,
					    channel, 0)).x;
  
  r2 -= read_imageui(image, sampler, (int4)(rCornerCoords.x-rParams0,
					    rCornerCoords.y+rParams1,
					    channel, 0)).x;
    
  r2 += read_imageui(image, sampler, (int4)(rCornerCoords.x+rParams0,
					    rCornerCoords.y+rParams1,
					    channel, 0)).x;
  
  r2 = (feat_t)round((float)r2/((rParams0*2+1)*(rParams1*2+1)));

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
