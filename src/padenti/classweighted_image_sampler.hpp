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

#ifndef __CLASSWEIGHTED_IMAGE_SAMPLER_HPP
#define __CLASSWEIGHTED_IMAGE_SAMPLER_HPP

template <typename type, unsigned int nChannels>
class ClassWeightedImageSampler: public ImageSampler<type, nChannels>
{
private:
  unsigned int m_nClasses;
  unsigned int *m_perClassBucketSize;
public:
  ClassWeightedImageSampler(unsigned int nSamples, unsigned int nClasses, unsigned int seed=0);
  ClassWeightedImageSampler(unsigned int nSamples, unsigned int nClasses, const float *weights, unsigned int seed=0);
  ~ClassWeightedImageSampler();
  unsigned int sample(const type *data, const unsigned char *labels,
		      unsigned int width, unsigned int height, unsigned int *samples) const;
};

#include <padenti/classweighted_image_sampler_impl.hpp>

#endif // __CLASSWEIGHTED_IMAGE_SAMPLER_HPP
