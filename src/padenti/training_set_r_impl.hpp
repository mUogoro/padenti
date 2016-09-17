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

#include <vector>
#include <utility>
#include <boost/log/trivial.hpp>
#include <padenti/training_set_r.hpp>

#define MAX_IMG_WIDTH (1920)
#define MAX_IMG_HEIGHT (1080)
#define MAX_IMG_CHANNELS (10)

template <typename type, unsigned int nChannels, unsigned int rSize>
TrainingSetR<type, nChannels, rSize>::TrainingSetR():
  m_nImages(0)
{}


template <typename type, unsigned int nChannels, unsigned int rSize>
TrainingSetR<type, nChannels, rSize>::TrainingSetR(const std::vector<std::string> &dataPaths,
						   ImageLoader<type, nChannels> &dataLoader,
						   const std::vector<std::vector<float> > &values,
						   const ImageSampler<type, nChannels> &sampler):
  m_nImages(0)
{
  /**
   * \todo how to deal with (rare) cases when image resolution or number of channels are greater than
   *       the ones supported?
   */
  type *imgBuff = new type[MAX_IMG_WIDTH*MAX_IMG_HEIGHT*MAX_IMG_CHANNELS];
  unsigned int imgWidth, imgHeight;
  unsigned int *samplesBuff = new unsigned int[MAX_IMG_WIDTH*MAX_IMG_HEIGHT];
  unsigned int nSamples;

  for (int i=0; i<dataPaths.size(); i++)
  {
    dataLoader.load(dataPaths.at(i), imgBuff, &imgWidth, &imgHeight);

    // Hack: create a mask where a value of 1 indicates that the first channel value is different
    // from zero
    /** \todo generalize to multichannel images */
    unsigned char *mask = new unsigned char[imgWidth*imgHeight];
    for (int j=0; j<imgWidth*imgHeight; j++) mask[j] = static_cast<unsigned char>(imgBuff[j]>0);

    nSamples = sampler.sample(imgBuff, mask, imgWidth, imgHeight,
			      samplesBuff);
    
    float value[rSize]; std::copy(values.at(i).begin(), values.at(i).end(), value);
    m_images.push_back(TrainingSetRImage<type, nChannels, rSize>(imgBuff, imgWidth, imgHeight,
								 value, samplesBuff, nSamples));

    delete []mask;
  }
  m_nImages = m_images.size();
  
  delete []samplesBuff;
  delete []imgBuff;
}

template <typename type, unsigned int nChannels, unsigned int rSize>
TrainingSetR<type, nChannels, rSize>::~TrainingSetR()
{}



template <typename type, unsigned int nChannels, unsigned int rSize>
const std::vector<TrainingSetRImage<type, nChannels, rSize> > &TrainingSetR<type, nChannels, rSize>::getImages() const
{
  return m_images;
}


template <typename type, unsigned int nChannels, unsigned int rSize>
TrainingSetR<type, nChannels, rSize>& TrainingSetR<type, nChannels, rSize>::operator<<(const TrainingSetRImage<type, nChannels, rSize> &image)
{
  m_images.push_back(image);
  m_nImages++;
}


template <typename type, unsigned int nChannels, unsigned int rSize>
unsigned int TrainingSetR<type, nChannels, rSize>::getNImages() const
{
  return m_nImages;
}
