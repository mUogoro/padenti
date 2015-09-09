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

#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/crc.hpp>
#include <padenti/uniform_image_sampler.hpp>


template <typename type, unsigned int nChannels>
UniformImageSampler<type, nChannels>::UniformImageSampler(unsigned int nSamples,
							  unsigned int seed):
  ImageSampler<type, nChannels>(nSamples, seed){}


template <typename type, unsigned int nChannels>
unsigned int UniformImageSampler<type, nChannels>::sample(const type *data, const unsigned char *labels,
							  unsigned int width, unsigned int height,
							  unsigned int *samples) const
{
  unsigned int *nonNullPixelsBuff = new unsigned int[width*height];
  unsigned int nonNullPixels=0;
  
  for (unsigned int id=0; id<width*height; id++)
  {
    if (labels[id]) nonNullPixelsBuff[nonNullPixels++]=id;
  }

  boost::random::mt19937 gen;
  boost::random::uniform_int_distribution<> dist(0, nonNullPixels-1);
  boost::crc_32_type checksum_agent;
  
  checksum_agent.process_bytes((unsigned char*)data, sizeof(type)*width*height*nChannels);
  gen.seed(checksum_agent.checksum() ^ this->m_seed);
  
  for (unsigned int i=0; i<this->m_nSamples; i++)
  {
    samples[i] = nonNullPixelsBuff[dist(gen)];
  }

  // Sort samples
  std::sort(samples, samples+this->m_nSamples);

  delete []nonNullPixelsBuff;
  return this->m_nSamples;
}
