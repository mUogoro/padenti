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

#ifndef __CL_IMG_FMT_TRAITS_HPP
#define __CL_IMG_FMT_TRAITS_HPP

#include <CL/cl.hpp>

template <typename ImgType, unsigned int nChannels>
struct ImgTypeTrait
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    throw "Unsupported type or number of channels";
  }
};


// Signed char
template <unsigned int nChannels>
struct ImgTypeTrait<char, nChannels>
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    clImgFmt.image_channel_order = CL_R;
    clImgFmt.image_channel_data_type = CL_SIGNED_INT8;
  }
};

template <>
void ImgTypeTrait<char, 2>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RG;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT8;
}

template <>
void ImgTypeTrait<char, 3>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGB;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT8;
}

template <>
void ImgTypeTrait<char, 4>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGBA;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT8;
}

// Unsigned char
template <unsigned int nChannels>
struct ImgTypeTrait<unsigned char, nChannels>
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    clImgFmt.image_channel_order = CL_R;
    clImgFmt.image_channel_data_type = CL_UNSIGNED_INT8;
  }
};

template <>
void ImgTypeTrait<unsigned char, 2>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RG;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT8;
}

template <>
void ImgTypeTrait<unsigned char, 3>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGB;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT8;
}

template <>
void ImgTypeTrait<unsigned char, 4>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGBA;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT8;
}



// Signed short int
template <unsigned int nChannels>
struct ImgTypeTrait<short int, nChannels>
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    clImgFmt.image_channel_order = CL_R;
    clImgFmt.image_channel_data_type = CL_SIGNED_INT16;
  }
};

template <>
void ImgTypeTrait<short int, 2>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RG;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT16;
}

template <>
void ImgTypeTrait<short int, 3>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGB;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT16;
}

template <>
void ImgTypeTrait<short int, 4>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGBA;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT16;
}


// Unsigned short int
template <unsigned int nChannels>
struct ImgTypeTrait<unsigned short int, nChannels>
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    clImgFmt.image_channel_order = CL_R;
    clImgFmt.image_channel_data_type = CL_UNSIGNED_INT16;
  }
};

template <>
void ImgTypeTrait<unsigned short int, 2>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RG;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT16;
}

template <>
void ImgTypeTrait<unsigned short int, 3>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGB;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT16;
}

template <>
void ImgTypeTrait<unsigned short int, 4>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGBA;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT16;
}


// Signed int
template <unsigned int nChannels>
struct ImgTypeTrait<int, nChannels>
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    clImgFmt.image_channel_order = CL_R;
    clImgFmt.image_channel_data_type = CL_SIGNED_INT32;
  }
};

template <>
void ImgTypeTrait<int, 2>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RG;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT32;
}

template <>
void ImgTypeTrait<int, 3>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGB;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT32;
}

template <>
void ImgTypeTrait<int, 4>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGBA;
  clImgFmt.image_channel_data_type = CL_SIGNED_INT32;
}


// Unsigned int
template <unsigned int nChannels>
struct ImgTypeTrait<unsigned int, nChannels>
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    clImgFmt.image_channel_order = CL_R;
    clImgFmt.image_channel_data_type = CL_UNSIGNED_INT32;
  }
};

template <>
void ImgTypeTrait<unsigned int, 2>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RG;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT32;
}

template <>
void ImgTypeTrait<unsigned int, 3>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGB;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT32;
}

template <>
void ImgTypeTrait<unsigned int, 4>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGBA;
  clImgFmt.image_channel_data_type = CL_UNSIGNED_INT32;
}


// Float
template <unsigned int nChannels>
struct ImgTypeTrait<float, nChannels>
{
public:
  static void toCLImgFmt(cl::ImageFormat &clImgFmt)
  {
    clImgFmt.image_channel_order = CL_R;
    clImgFmt.image_channel_data_type = CL_FLOAT;
  }
};

template <>
void ImgTypeTrait<float, 1>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_R;
  clImgFmt.image_channel_data_type = CL_FLOAT;
}

template <>
void ImgTypeTrait<float, 2>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RG;
  clImgFmt.image_channel_data_type = CL_FLOAT;
}

template <>
void ImgTypeTrait<float, 3>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGB;
  clImgFmt.image_channel_data_type = CL_FLOAT;
}

template <>
void ImgTypeTrait<float, 4>::toCLImgFmt(cl::ImageFormat &clImgFmt)
{
  clImgFmt.image_channel_order = CL_RGBA;
  clImgFmt.image_channel_data_type = CL_FLOAT;
}



#endif // __CL_IMG_FMT_TRAITS_HPP
