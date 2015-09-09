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
#include <boost/filesystem.hpp>
#include <padenti/training_set.hpp>

using namespace boost::filesystem;

#define MAX_IMG_WIDTH (1920)
#define MAX_IMG_HEIGHT (1080)
#define MAX_IMG_CHANNELS (10)

template <typename type, unsigned int nChannels>
TrainingSet<type, nChannels>::TrainingSet(const std::string &tsPathStr,
					  const std::string &dataSuffix, const std::string &labelsSuffix,
					  unsigned int nClasses,
					  ImageLoader<type, nChannels> &dataLoader,
					  ImageLoader<unsigned char, 1> &labelsLoader,
					  const ImageSampler<type, nChannels> &sampler):
  m_nClasses(nClasses),
  m_tsPath(tsPathStr)
{
  std::vector<std::pair<std::string, std::string> > imgLabelsPairs;
  path tsPath(tsPathStr);

  directory_iterator endItr;
  for (directory_iterator itr(tsPath); itr!=endItr; ++itr)
  {
    if (!is_directory(itr->status()))
    {
      // Look if the current file is a training set sample
      std::string fName = itr->path().filename().string();
      unsigned int prefixLen = fName.size()-dataSuffix.size();
      if (fName.rfind(dataSuffix)==prefixLen)
      {
	// Sample found!!! Look for the labels image
	std::string labelsImgName = tsPathStr + "/" + fName.substr(0, prefixLen) + labelsSuffix;
	if (exists(labelsImgName))
	{
	  fName = tsPathStr + "/" + fName;
	  imgLabelsPairs.push_back(std::make_pair(fName, labelsImgName));
	}
      }
    }
  }

  /**
   * \todo how to deal with (rare) cases when image resolution or number of channels are greater than
   *       the ones supported?
   */
  type *imgBuff = new type[MAX_IMG_WIDTH*MAX_IMG_HEIGHT*MAX_IMG_CHANNELS];
  unsigned int imgWidth, imgHeight;
  unsigned char *labelsBuff = new unsigned char[MAX_IMG_WIDTH*MAX_IMG_HEIGHT];
  unsigned int labelsWidth, labelsHeight;
  unsigned int *samplesBuff = new unsigned int[MAX_IMG_WIDTH*MAX_IMG_HEIGHT];
  unsigned int nSamples;


  for (std::vector<std::pair<std::string, std::string> >::iterator it = imgLabelsPairs.begin();
       it!=imgLabelsPairs.end(); ++it)
  {
    std::string imgFName = it->first;
    std::string labelsFName = it->second;

    dataLoader.load(imgFName, imgBuff, &imgWidth, &imgHeight);
    labelsLoader.load(labelsFName, labelsBuff, &labelsWidth, &labelsHeight);

    /** \todo log skipped samples? */
    if (imgWidth!=labelsWidth || imgHeight!=labelsHeight) continue;

    nSamples = sampler.sample(imgBuff, labelsBuff, imgWidth, imgHeight,
			      samplesBuff);
    
    m_images.push_back(TrainingSetImage<type, nChannels>(imgBuff, imgWidth, imgHeight,
							 labelsBuff, m_nClasses,
							 samplesBuff, nSamples));
  }
  m_nImages = m_images.size();
  
  /** Compute per-class priors */
  m_priors = new float[m_nClasses];
  std::fill_n(m_priors, m_nClasses, 0.0f);

  for (typename std::vector<TrainingSetImage<type, nChannels> >::const_iterator it=m_images.begin();
       it!=m_images.end(); ++it)
  {
    const float *priors = it->getPriors();
    for (unsigned int i=0; i<m_nClasses; i++)
    {
      m_priors[i] += priors[i];
    }
  }
  for (int i=0; i<m_nClasses; i++)
  {
    m_priors[i] *= 1.0f/(float)m_nImages;
  }
  

  delete []samplesBuff;
  delete []labelsBuff;
  delete []imgBuff;
}


template <typename type, unsigned int nChannels>
TrainingSet<type, nChannels>::~TrainingSet()
{
  delete []m_priors;
}


template <typename type, unsigned int nChannels>
const std::vector<TrainingSetImage<type, nChannels> > &TrainingSet<type, nChannels>::getImages() const
{
  return m_images;
}


template <typename type, unsigned int nChannels>
unsigned int TrainingSet<type, nChannels>::getNImages() const
{
  return m_nImages;
}


template <typename type, unsigned int nChannels>
const float* TrainingSet<type, nChannels>::getPriors() const
{
  return m_priors;
}
