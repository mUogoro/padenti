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

//#include <prng.cl>
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

uint4 md5Rand(uint4 seed);
__kernel void _computePerImageHistogram(__read_only image_t image,
				       uint nChannels, uint width, uint height,
				       __read_only image2d_t labels,
				       __read_only image2d_t nodesID,
				       __global uint *samples, uint nSamples,
				       uint featDim,
				       __global feat_t *featLowBounds, __global feat_t *featUpBounds,
				       uint nThresholds, feat_t thrLowBound, feat_t thrUpBound,
				       __global uchar *perImageHistogram,
				       uint treeID, int startNode, int endNode,
				       __global int *treeLeftChildren,
				       __global float *treePosteriors,
				       __local feat_t *tmp,
				       __local feat_t *featuresBuff)
                                       //,uint baseSeed)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  uint4 seed;
  feat_t feat, thr;
  int2 coords;
  int nodeID;

  coords.x = samples[get_global_id(0)];
  coords = (int2)(coords.x%width, coords.x/width);
  nodeID = read_imagei(nodesID, sampler, coords).x;

  seed.x = treeID;
  seed.y = nodeID;
  seed.z = get_global_id(1);
  seed.w = 0; // Non zero value for 4th int???

  /*
  seed.x = treeID^baseSeed;
  seed.y = nodeID^baseSeed;
  seed.z = get_global_id(1)^baseSeed;
  seed.w = baseSeed;
  */

  /** 
   * \todo better int32-to-float32 conversion
  */
  for (int i=0; i<featDim; i+=4)
  {
    seed = md5Rand(seed);

    // The first 8 work-items of each work-group copy the upper/lower feature bounds
    // to shared in order to avoid multiple reads from global memory
    int gtThrID = min(4, ((int)featDim)-i);
    if (!get_local_id(0)&&get_local_id(1)<gtThrID*2)
    {
      tmp[get_local_id(1)] = \
	(get_local_id(1)<gtThrID) ? featLowBounds[i+get_local_id(1)] : featUpBounds[i+get_local_id(1)-gtThrID];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    

    /** \todo  Find another way the uint32-to-float conversion since uncorrect results get
     are returned by old fglrx driver versions (e.g. default Ubuntu 14.04 version) */
    feat = tmp[0] + 
      (feat_t)(((float)(seed.x))/(0xFFFFFFFF)*(tmp[gtThrID]-tmp[0]));
    ACCESS_FEATURE(featuresBuff, i, featDim) = feat;
    if ((i+1)>=featDim) break;
    
    feat = tmp[1] + 
      (feat_t)(((float)seed.y)/(0xFFFFFFFF)*(tmp[gtThrID+1]-tmp[1]));
    ACCESS_FEATURE(featuresBuff, i+1, featDim) = feat;
    if ((i+2)>=featDim) break;
    
    feat = tmp[2] +
      (feat_t)(((float)seed.z)/(0xFFFFFFFF)*(tmp[gtThrID+2]-tmp[2]));
    ACCESS_FEATURE(featuresBuff, i+2, featDim) = feat;
    if ((i+3)>=featDim) break;

    feat = tmp[3] +
      (feat_t)(((float)seed.w)/(0xFFFFFFFF)*(tmp[gtThrID+3]-tmp[3]));
    ACCESS_FEATURE(featuresBuff, i+3, featDim) = feat;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Perform computation on image samples only if they reach the current slice of
  // trained leaves
  if (nodeID>=startNode && nodeID<=endNode)
  {
    feat = computeFeature(image, nChannels, width, height, coords,
			  treeLeftChildren, treePosteriors, nodesID,
			  featuresBuff, featDim);
  
    /** \todo speed up threshold sampling by sampling 4 thresholds at a time */
    seed.x = treeID;
    seed.y = read_imagei(nodesID, sampler, coords).x;
    seed.z = get_global_id(1);
    seed.w = 1;
    perImageHistogram += get_global_id(0)*nThresholds*get_global_size(1)+get_global_id(1);
        
    for (uint t=0; t<nThresholds;)
    { 
      seed = md5Rand(seed);

      thr = thrLowBound + 
	(feat_t)(((float)seed.x)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
      if ((t)>=nThresholds) break;

      thr = thrLowBound + 
	(feat_t)(((float)seed.y)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
      if ((t)>=nThresholds) break;
    
      thr = thrLowBound + 
	(feat_t)(((float)seed.z)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
      if ((t)>=nThresholds) break;
      
      thr = thrLowBound + 
	(feat_t)(((float)seed.w)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
    }
    
  }
}


/** \todo: remove unused arguments */
__kernel void computePerImageHistogramWithLUT(__read_only image_t image,
					      uint nChannels, uint width, uint height,
					      __read_only image2d_t labels,
					      __read_only image2d_t nodesID,
					      __global uint *samples, uint nSamples,
					     uint featDim,
					      __global feat_t *featLowBounds, __global feat_t *featUpBounds,
					      uint nThresholds, feat_t thrLowBound, feat_t thrUpBound,
					      __global uchar *perImageHistogram,
					      uint treeID, int startNode, int endNode,
					     __global int *treeLeftChildren,
					      __global float *treePosteriors,
					      __local feat_t *tmp,
					      __local feat_t *featuresBuff,
					      __global feat_t *lut, uint lutSize)
                                              //,uint baseSeed)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  uint4 seed;
  feat_t feat, thr;
  int2 coords;
  int nodeID;

  coords.x = samples[get_global_id(0)];
  coords = (int2)(coords.x%width, coords.x/width);
  nodeID = read_imagei(nodesID, sampler, coords).x;

  seed.x = treeID;
  seed.y = nodeID;
  seed.z = get_global_id(1);
  seed.w = 0; // Non zero value for 4th int???

  /*
  seed.x = treeID^baseSeed;
  seed.y = nodeID^baseSeed;
  seed.z = get_global_id(1)^baseSeed;
  seed.w = baseSeed;
  */

  /** \todo: avoid waste of resource: only one int32 out of four is used. */
  seed = md5Rand(seed);

  /** 
   * \todo better int32-to-float32 conversion
  */
  int idx = round((float)seed.x/0xFFFFFFFF * lutSize)*featDim;

  for (int i=0; i<featDim; i++)
  {
    ACCESS_FEATURE(featuresBuff, i, featDim) = lut[idx+i];
  }

  // Perform computation on image samples only if they reach the current slice of
  // trained leaves
  if (nodeID>=startNode && nodeID<=endNode)
  {

    //offset = (get_global_id(0)*get_global_size(1)+get_global_id(1))*featDim;
    feat = computeFeature(image, nChannels, width, height, coords,
			  treeLeftChildren, treePosteriors, nodesID,
			  featuresBuff, featDim);
  
    /** \todo speed up threshold sampling by sampling 4 thresholds at a time */
    seed.x = treeID;
    seed.y = read_imagei(nodesID, sampler, coords).x;
    seed.z = get_global_id(1);
    seed.w = 1;
    perImageHistogram += get_global_id(0)*nThresholds*get_global_size(1)+get_global_id(1);
    
    for (uint t=0; t<nThresholds;)
    { 
      seed = md5Rand(seed);

      thr = thrLowBound + 
	(feat_t)(((float)seed.x)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
      if ((t)>=nThresholds) break;

      thr = thrLowBound + 
	(feat_t)(((float)seed.y)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
      if ((t)>=nThresholds) break;
    
      thr = thrLowBound + 
	(feat_t)(((float)seed.z)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
      if ((t)>=nThresholds) break;
      
      thr = thrLowBound + 
	(feat_t)(((float)seed.w)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
      *perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
      perImageHistogram += get_global_size(1);
      ++t;
    }
    
  }
}


uint4 md5Rand(uint4 seed);
__kernel void computePerImageHistogram(__read_only image_t image,
				       uint nChannels, uint width, uint height,
				       __read_only image2d_t labels,
				       __read_only image2d_t nodesID,
				       __global uint *samples, uint nSamples,
				       uint featDim,
				       __global feat_t *featLowBounds, __global feat_t *featUpBounds,
				       uint nThresholds, feat_t thrLowBound, feat_t thrUpBound,
				       __global uchar *perImageHistogram,
				       uint treeID, int startNode, int endNode,
				       __global int *treeLeftChildren,
				       __global float *treePosteriors,
				       __global feat_t *featLut, uint featLutSize,
				       __global feat_t *thrLut, uint thrLutSize,
				       __local feat_t *tmp,
				       __local feat_t *featuresBuff)
                                       //,uint baseSeed)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  uint4 seed;
  feat_t feat, thr;
  int2 coords;
  int nodeID;

  coords.x = samples[get_global_id(0)];
  coords = (int2)(coords.x%width, coords.x/width);
  nodeID = read_imagei(nodesID, sampler, coords).x;

  seed.x = treeID;
  seed.y = nodeID;
  seed.z = get_global_id(1);
  seed.w = 0; // Non zero value for 4th int???

  /*
  seed.x = treeID^baseSeed;
  seed.y = nodeID^baseSeed;
  seed.z = get_global_id(1)^baseSeed;
  seed.w = baseSeed;
  */

  if (featLutSize)
  {
    /** \todo: avoid waste of resource: only one int32 out of four is used. */
    seed = md5Rand(seed);

    /** 
     * \todo better int32-to-float32 conversion
     */
    int idx = floor((float)seed.x/0xFFFFFFFF * featLutSize)*featDim;

    for (int i=0; i<featDim; i++)
    {
      ACCESS_FEATURE(featuresBuff, i, featDim) = featLut[idx+i];
    }
  }

  else
  {
    for (int i=0; i<featDim; i+=4)
    {
      seed = md5Rand(seed);

      // The first 8 work-items of each work-group copy the upper/lower feature bounds
      // to shared in order to avoid multiple reads from global memory
      int gtThrID = min(4, ((int)featDim)-i);
      if (!get_local_id(0)&&get_local_id(1)<gtThrID*2)
      {
	tmp[get_local_id(1)] =						\
	  (get_local_id(1)<gtThrID) ? featLowBounds[i+get_local_id(1)] : featUpBounds[i+get_local_id(1)-gtThrID];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    

      /** \todo  Find another way the uint32-to-float conversion since uncorrect results get
	  are returned by old fglrx driver versions (e.g. default Ubuntu 14.04 version) */
      feat = tmp[0] + 
	(feat_t)(((float)(seed.x))/(0xFFFFFFFF)*(tmp[gtThrID]-tmp[0]));
      ACCESS_FEATURE(featuresBuff, i, featDim) = feat;
      if ((i+1)>=featDim) break;
    
      feat = tmp[1] + 
	(feat_t)(((float)seed.y)/(0xFFFFFFFF)*(tmp[gtThrID+1]-tmp[1]));
      ACCESS_FEATURE(featuresBuff, i+1, featDim) = feat;
      if ((i+2)>=featDim) break;
    
      feat = tmp[2] +
	(feat_t)(((float)seed.z)/(0xFFFFFFFF)*(tmp[gtThrID+2]-tmp[2]));
      ACCESS_FEATURE(featuresBuff, i+2, featDim) = feat;
      if ((i+3)>=featDim) break;

      feat = tmp[3] +
	(feat_t)(((float)seed.w)/(0xFFFFFFFF)*(tmp[gtThrID+3]-tmp[3]));
      ACCESS_FEATURE(featuresBuff, i+3, featDim) = feat;

      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  // Perform computation on image samples only if they reach the current slice of
  // trained leaves
  if (nodeID>=startNode && nodeID<=endNode)
  {
    feat = computeFeature(image, nChannels, width, height, coords,
			  treeLeftChildren, treePosteriors, nodesID,
			  featuresBuff, featDim);
  
    /** \todo speed up threshold sampling by sampling 4 thresholds at a time */
    seed.x = treeID;
    seed.y = read_imagei(nodesID, sampler, coords).x;
    seed.z = get_global_id(1);
    seed.w = 1;
    perImageHistogram += get_global_id(0)*nThresholds*get_global_size(1)+get_global_id(1);
    
    if (thrLutSize)
    {
      for (uint t=0; t<nThresholds;)
      { 
	seed = md5Rand(seed);

	int idx = floor((float)seed.x/0xFFFFFFFF * thrLutSize);
	thr = thrLut[idx];
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
	if ((t)>=nThresholds) break;

	idx = floor((float)seed.y/0xFFFFFFFF * thrLutSize);
	thr = thrLut[idx];
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
	if ((t)>=nThresholds) break;
    
	idx = floor((float)seed.z/0xFFFFFFFF * thrLutSize);
	thr = thrLut[idx];
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
	if ((t)>=nThresholds) break;
      
	idx = floor((float)seed.w/0xFFFFFFFF * thrLutSize);
	thr = thrLut[idx];
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
      }
    }
    else
    {
      for (uint t=0; t<nThresholds;)
      { 
	seed = md5Rand(seed);

	thr = thrLowBound + 
	  (feat_t)(((float)seed.x)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
	if ((t)>=nThresholds) break;

	thr = thrLowBound + 
	  (feat_t)(((float)seed.y)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
	if ((t)>=nThresholds) break;
    
	thr = thrLowBound + 
	  (feat_t)(((float)seed.z)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
	if ((t)>=nThresholds) break;
      
	thr = thrLowBound + 
	  (feat_t)(((float)seed.w)/(0xFFFFFFFF)*(thrUpBound-thrLowBound));
	*perImageHistogram = (uchar)((feat<=thr) ? 1 : 0);
	perImageHistogram += get_global_size(1);
	++t;
      }
    }
  }
}




/**********************************************************/
// MD5-based PRNG
/**********************************************************/
// Round rotations
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21


#define ROTATE_LEFT(x, n) (((x)<<(n))|((x)>>(32-(n))))

// Non-linear functions
#define F(x, y, z) (((x)&(y))|((~x)&(z)))
#define G(x, y, z) (((x)&(z))|((y)&(~z)))
#define H(x, y, z) ((x)^(y)^(z))
#define I(x, y, z) ((y)^((x)|(~z)))


// Round functions
// - a,b,c,d 32bit words of state
// - x passpharase value
// - s bits of rotation
// - ac round constant
#define FF(a, b, c, d, x, s, ac) {                      \
    (a) += F((b), (c), (d))+(x)+(unsigned int)(ac);     \
    (a) = ROTATE_LEFT((a), (s));                        \
    (a) += (b);                                         \
  }

#define GG(a, b, c, d, x, s, ac) {                      \
    (a) += G ((b), (c), (d))+(x)+(unsigned int)(ac);    \
    (a) = ROTATE_LEFT ((a), (s));                       \
    (a) += (b);                                         \
  }

#define HH(a, b, c, d, x, s, ac) {                      \
    (a) += H ((b), (c), (d))+(x)+(unsigned int)(ac);    \
    (a) = ROTATE_LEFT ((a), (s));                       \
    (a) += (b);                                         \
  }

#define II(a, b, c, d, x, s, ac) {                      \
    (a) += I ((b), (c), (d))+(x)+(unsigned int)(ac);    \
    (a) = ROTATE_LEFT ((a), (s));                       \
    (a) += (b);                                         \
  }

// Optimized round function, called when message's words value
// is zero: this allows to avoid one addition
#define FF_noadd(a, b, c, d, x, s, ac) {                        \
    (a) += F((b), (c), (d))+(unsigned int)(ac); \
    (a) = ROTATE_LEFT((a), (s));                        \
    (a) += (b);                                         \
  }

#define GG_noadd(a, b, c, d, x, s, ac) {                        \
    (a) += G ((b), (c), (d))+(unsigned int)(ac);        \
    (a) = ROTATE_LEFT ((a), (s));                       \
    (a) += (b);                                         \
  }

#define HH_noadd(a, b, c, d, x, s, ac) {                        \
    (a) += H ((b), (c), (d))+(unsigned int)(ac);        \
    (a) = ROTATE_LEFT ((a), (s));                       \
    (a) += (b);                                         \
  }

#define II_noadd(a, b, c, d, x, s, ac) {                        \
    (a) += I ((b), (c), (d))+(unsigned int)(ac);        \
    (a) = ROTATE_LEFT ((a), (s));                       \
    (a) += (b);                                         \
  }



uint4 md5Rand(uint4 seed)
{
  uint4 state;

  // Init state
  state.x = 0x67452301;
  state.y = 0xefcdab89;
  state.z = 0x98badcfe;
  state.w = 0x10325476;

  
  // Perform rounds
  // Round 1
  FF(state.x, state.y, state.z, state.w, seed.x, S11, 0xd76aa478); // 1
  FF(state.w, state.x, state.y, state.z, seed.y, S12, 0xe8c7b756); // 2 
  FF(state.z, state.w, state.x, state.y, seed.z, S13, 0x242070db); // 3 
  FF(state.y, state.z, state.w, state.x, seed.w, S14, 0xc1bdceee); // 4 
  FF(state.x, state.y, state.z, state.w, 128, S11, 0xf57c0faf); // 5 
  FF_noadd(state.w, state.x, state.y, state.z, 0x00, S12, 0x4787c62a); // 6 
  FF_noadd(state.z, state.w, state.x, state.y, 0x00, S13, 0xa8304613); // 7 
  FF_noadd(state.y, state.z, state.w, state.x, 0x00, S14, 0xfd469501); // 8 
  FF_noadd(state.x, state.y, state.z, state.w, 0x00, S11, 0x698098d8); // 9 
  FF_noadd(state.w, state.x, state.y, state.z, 0x00, S12, 0x8b44f7af); // 10 
  FF_noadd(state.z, state.w, state.x, state.y, 0x00, S13, 0xffff5bb1); // 11 
  FF_noadd(state.y, state.z, state.w, state.x, 0x00, S14, 0x895cd7be); // 12 
  FF_noadd(state.x, state.y, state.z, state.w, 0x00, S11, 0x6b901122); // 13 
  FF_noadd(state.w, state.x, state.y, state.z, 0x00, S12, 0xfd987193); // 14 
  FF(state.z, state.w, state.x, state.y, 128, S13, 0xa679438e); // 15 
  FF_noadd(state.y, state.z, state.w, state.x, 0x00, S14, 0x49b40821); // 16 

  // Round 2
  GG(state.x, state.y, state.z, state.w, seed.y, S21, 0xf61e2562); // 17 
  GG_noadd(state.w, state.x, state.y, state.z, 0x00, S22, 0xc040b340); // 18 
  GG_noadd(state.z, state.w, state.x, state.y, 0x00, S23, 0x265e5a51); // 19 
  GG(state.y, state.z, state.w, state.x, seed.x, S24, 0xe9b6c7aa); // 20 
  GG_noadd(state.x, state.y, state.z, state.w, 0x00, S21, 0xd62f105d); // 21 
  GG_noadd(state.w, state.x, state.y, state.z, 0x00, S22,  0x2441453); // 22 
  GG_noadd(state.z, state.w, state.x, state.y, 0x00, S23, 0xd8a1e681); // 23 
  GG(state.y, state.z, state.w, state.x, 128, S24, 0xe7d3fbc8); // 24 
  GG_noadd(state.x, state.y, state.z, state.w, 0x00, S21, 0x21e1cde6); // 25 
  GG(state.w, state.x, state.y, state.z, 128, S22, 0xc33707d6); // 26 
  GG(state.z, state.w, state.x, state.y, seed.w, S23, 0xf4d50d87); // 27 
  GG_noadd(state.y, state.z, state.w, state.x, 0x00, S24, 0x455a14ed); // 28 
  GG_noadd(state.x, state.y, state.z, state.w, 0x00, S21, 0xa9e3e905); // 29 
  GG(state.w, state.x, state.y, state.z, seed.z, S22, 0xfcefa3f8); // 30 
  GG_noadd(state.z, state.w, state.x, state.y, 0x00, S23, 0x676f02d9); // 31 
  GG_noadd(state.y, state.z, state.w, state.x, 0x00, S24, 0x8d2a4c8a); // 32 

  // Round 3
  HH_noadd(state.x, state.y, state.z, state.w, 0x00, S31, 0xfffa3942); // 33 
  HH_noadd(state.w, state.x, state.y, state.z, 0x00, S32, 0x8771f681); // 34 
  HH_noadd(state.z, state.w, state.x, state.y, 0x00, S33, 0x6d9d6122); // 35 
  HH(state.y, state.z, state.w, state.x, 128, S34, 0xfde5380c); // 36 
  HH(state.x, state.y, state.z, state.w, seed.y, S31, 0xa4beea44); // 37 
  HH(state.w, state.x, state.y, state.z, 128, S32, 0x4bdecfa9); // 38 
  HH_noadd(state.z, state.w, state.x, state.y, 0x00, S33, 0xf6bb4b60); // 39 
  HH_noadd(state.y, state.z, state.w, state.x, 0x00, S34, 0xbebfbc70); // 40 
  HH_noadd(state.x, state.y, state.z, state.w, 0x00, S31, 0x289b7ec6); // 41 
  HH(state.w, state.x, state.y, state.z, seed.x, S32, 0xeaa127fa); // 42 
  HH(state.z, state.w, state.x, state.y, seed.w, S33, 0xd4ef3085); // 43 
  HH_noadd(state.y, state.z, state.w, state.x, 0x00, S34,  0x4881d05); // 44 
  HH_noadd(state.x, state.y, state.z, state.w, 0x00, S31, 0xd9d4d039); // 45 
  HH_noadd(state.w, state.x, state.y, state.z, 0x00, S32, 0xe6db99e5); // 46 
  HH_noadd(state.z, state.w, state.x, state.y, 0x00, S33, 0x1fa27cf8); // 47 
  HH(state.y, state.z, state.w, state.x, seed.z, S34, 0xc4ac5665); // 48 

  // Round 4
  II(state.x, state.y, state.z, state.w, seed.x, S41, 0xf4292244); // 49 
  II_noadd(state.w, state.x, state.y, state.z, 0x00, S42, 0x432aff97); // 50 
  II(state.z, state.w, state.x, state.y, 128, S43, 0xab9423a7); // 51 
  II_noadd(state.y, state.z, state.w, state.x, 0x00, S44, 0xfc93a039); // 52 
  II_noadd(state.x, state.y, state.z, state.w, 0x00, S41, 0x655b59c3); // 53 
  II(state.w, state.x, state.y, state.z, seed.w, S42, 0x8f0ccc92); // 54 
  II_noadd(state.z, state.w, state.x, state.y, 0x00, S43, 0xffeff47d); // 55 
  II(state.y, state.z, state.w, state.x, seed.y, S44, 0x85845dd1); // 56 
  II_noadd(state.x, state.y, state.z, state.w, 0x00, S41, 0x6fa87e4f); // 57 
  II_noadd(state.w, state.x, state.y, state.z, 0x00, S42, 0xfe2ce6e0); // 58 
  II_noadd(state.z, state.w, state.x, state.y, 0x00, S43, 0xa3014314); // 59 
  II_noadd(state.y, state.z, state.w, state.x, 0x00, S44, 0x4e0811a1); // 60 
  II(state.x, state.y, state.z, state.w, 128, S41, 0xf7537e82); // 61 
  II_noadd(state.w, state.x, state.y, state.z, 0x00, S42, 0xbd3af235); // 62 
  II(state.z, state.w, state.x, state.y, seed.z, S43, 0x2ad7d2bb); // 63 
  II_noadd(state.y, state.z, state.w, state.x, 0x00, S44, 0xeb86d391); // 64 
  
  state.x += 0x67452301;
  state.y += 0xEFCDAB89;
  state.z += 0x98BADCFE;
  state.w += 0x10325476;

  // Done
  return state;
}
