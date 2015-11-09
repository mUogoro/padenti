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
#ifndef __PADENTI_BASE_HPP
#define __PADENTI_BASE_HPP

// Define the supported image and feature type, for example
// - (integral) int images with 4 channels
// - float features of size 10
// Define the number of classes as well
//typedef unsigned int ImgType;
//static const int N_CHANNELS = 4;
//typedef float FeatType;
//static const int FEAT_SIZE = 10;
//static const int N_CLASSES = 21;

typedef unsigned int ImgType;
static const int N_CHANNELS = 10;
typedef short int FeatType;
static const int FEAT_SIZE = 10;
static const int N_CLASSES = 2;

#endif // __PADENTI_BASE_HPP