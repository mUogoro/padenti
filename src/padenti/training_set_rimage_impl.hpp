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
#include <padenti/training_set_rimage.hpp>

template <typename type, unsigned int nChannels, unsigned int rSize>
TrainingSetRImage<type, nChannels, rSize>::TrainingSetRImage(const type *data, unsigned int width, unsigned int height,
							     const float *value,
							     const unsigned int *samples, unsigned int nSamples):
  Image<type, nChannels>(data, width, height), m_nSamples(nSamples)
{
  m_samples = new unsigned int[m_nSamples];
  std::copy(samples, samples+m_nSamples, m_samples);
  std::copy(value, value+rSize, m_value);
}


template <typename type, unsigned int nChannels, unsigned int rSize>
TrainingSetRImage<type, nChannels, rSize>::TrainingSetRImage(const TrainingSetRImage<type, nChannels, rSize> &tsImg):
  Image<type, nChannels>(tsImg),
  m_nSamples(tsImg.m_nSamples)
{
  m_samples = new unsigned int[m_nSamples];
  std::copy(tsImg.m_samples, tsImg.m_samples+tsImg.m_nSamples, m_samples);
  std::copy(tsImg.m_value, tsImg.m_value+rSize, m_value);
}


template <typename type, unsigned int nChannels, unsigned int rSize>
TrainingSetRImage<type, nChannels, rSize>::~TrainingSetRImage()
{
  delete []m_samples;
}


template <typename type, unsigned int nChannels, unsigned int rSize>
const float *TrainingSetRImage<type, nChannels, rSize>::getValue() const
{
  return m_value;
}

template <typename type, unsigned int nChannels, unsigned int rSize>
const unsigned int *TrainingSetRImage<type, nChannels, rSize>::getSamples() const
{
  return m_samples;
}

template <typename type, unsigned int nChannels, unsigned int rSize>
unsigned int TrainingSetRImage<type, nChannels, rSize>::getNSamples() const
{
  return m_nSamples;
}
