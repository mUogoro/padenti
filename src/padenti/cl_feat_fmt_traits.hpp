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

#ifndef __CL_FEAT_FMT_TRAITS_HPP
#define __CL_FEAT_FMT_TRAITS_HPP

#include <CL/cl.hpp>

template <typename FeatType>
struct FeatTypeTrait
{
public:
  static void getCLTypedefCode(std::string &code)
  {
    throw "Unsupported feature type";
  }
};


// Signed char
template <>
void FeatTypeTrait<char>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef char feat_t;\n\n#endif\n");
}

// Unsigned char
template <>
void FeatTypeTrait<unsigned char>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef uchar feat_t;\n\n#endif\n");
}


// Signed short int
template <>
void FeatTypeTrait<short>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef short feat_t;\n\n#endif\n");
}

// Unsigned short int
template <>
void FeatTypeTrait<unsigned short>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef ushort feat_t;\n\n#endif\n");
}


// Signed int
template <>
void FeatTypeTrait<int>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef int feat_t;\n\n#endif\n");
}

// Unigned int
template <>
void FeatTypeTrait<unsigned int>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef uint feat_t;\n\n#endif\n");
}


// Long int
template <>
void FeatTypeTrait<long>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef long feat_t;\n\n#endif\n");
}

// Unsigned long int
template <>
void FeatTypeTrait<unsigned long>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef ulong feat_t;\n\n#endif\n");
}


// Float
template <>
void FeatTypeTrait<float>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef float feat_t;\n\n#endif\n");
}


// Double
template <>
void FeatTypeTrait<double>::getCLTypedefCode(std::string &code)
{
  code.assign("#ifndef __FEAT_TYPE\n#define __FEAT_TYPE\n\ntypedef double feat_t;\n\n#endif\n");
}

#endif // __CL_FEAT_FMT_TRAITS_HPP
