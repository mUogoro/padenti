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
#include <padenti/training_set_image.hpp>

template <typename type, unsigned int nChannels>
TrainingSetImage<type, nChannels>::TrainingSetImage(const type *data, unsigned int width, unsigned int height,
						    const unsigned char *labels, unsigned int nClasses,
						    const unsigned int *samples, unsigned int nSamples):
  Image<type, nChannels>(data, width, height),  m_nClasses(nClasses), m_nSamples(nSamples)
{
  unsigned int totLabeledSamples;
  unsigned int *perClassSamples = new unsigned int[m_nClasses];

  m_priors = new float[m_nClasses];
  std::fill_n(m_priors, m_nClasses, 0.0f);

  // Compute the per-image/per-class priors on labeled sampled pixels
  /** \todo how to handle unlabeled pixels? (e.g. for transductive learning) */
  /** \todo compute real-priors as weel, i.e. priors on the whole pixels set */
  totLabeledSamples = 0;
  std::fill_n(perClassSamples, m_nClasses, 0);
  for (int i=0; i<nSamples; i++)
  {
    if (labels[samples[i]])
    {
      perClassSamples[labels[samples[i]]-1]++;
      totLabeledSamples++;
    }
  }

  for (int i=0; i<m_nClasses; i++)
  {
    m_priors[i] = (float)perClassSamples[i]/totLabeledSamples;
  }

  m_samples = new unsigned int[m_nSamples];
  m_labels = new unsigned char[this->m_width*this->m_height];
  std::copy(labels, labels+this->m_width*this->m_height, m_labels);
  std::copy(samples, samples+m_nSamples, m_samples);
  
  delete []perClassSamples;
}


template <typename type, unsigned int nChannels>
TrainingSetImage<type, nChannels>::TrainingSetImage(const TrainingSetImage<type, nChannels> &tsImg):
  Image<type, nChannels>(tsImg),
  m_nSamples(tsImg.m_nSamples),
  m_nClasses(tsImg.m_nClasses)
{
  m_samples = new unsigned int[m_nSamples];
  std::copy(tsImg.m_samples, tsImg.m_samples+tsImg.m_nSamples, m_samples);

  m_labels = new unsigned char[this->m_width*this->m_height];
  std::copy(tsImg.m_labels, tsImg.m_labels+tsImg.m_width*tsImg.m_height, m_labels);

  m_priors = new float[m_nClasses];
  std::copy(tsImg.m_priors, tsImg.m_priors+tsImg.m_nClasses, m_priors);
}


template <typename type, unsigned int nChannels>
TrainingSetImage<type, nChannels>::~TrainingSetImage()
{
  delete []m_priors;
  delete []m_labels;
  delete []m_samples;
}



template <typename type, unsigned int nChannels>
const unsigned char *TrainingSetImage<type, nChannels>::getLabels() const
{
  return m_labels;
}

template <typename type, unsigned int nChannels>
const unsigned int *TrainingSetImage<type, nChannels>::getSamples() const
{
  return m_samples;
}

template <typename type, unsigned int nChannels>
unsigned int TrainingSetImage<type, nChannels>::getNSamples() const
{
  return m_nSamples;
}

template <typename type, unsigned int nChannels>
const float *TrainingSetImage<type, nChannels>::getPriors() const
{
  return m_priors;
}
