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

#ifndef __PRNG_HPP
#define __PRNG_HPP

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


void md5Rand(const unsigned int seed[], unsigned int state[])
{
  // Init state
  state[0] = 0x67452301;
  state[1] = 0xefcdab89;
  state[2] = 0x98badcfe;
  state[3] = 0x10325476;

  
  // Perform rounds
  // Round 1
  FF(state[0], state[1], state[2], state[3], seed[0], S11, 0xd76aa478); // 1
  FF(state[3], state[0], state[1], state[2], seed[1], S12, 0xe8c7b756); // 2 
  FF(state[2], state[3], state[0], state[1], seed[2], S13, 0x242070db); // 3 
  FF(state[1], state[2], state[3], state[0], seed[3], S14, 0xc1bdceee); // 4 
  FF(state[0], state[1], state[2], state[3], 128, S11, 0xf57c0faf); // 5 
  FF_noadd(state[3], state[0], state[1], state[2], 0x00, S12, 0x4787c62a); // 6 
  FF_noadd(state[2], state[3], state[0], state[1], 0x00, S13, 0xa8304613); // 7 
  FF_noadd(state[1], state[2], state[3], state[0], 0x00, S14, 0xfd469501); // 8 
  FF_noadd(state[0], state[1], state[2], state[3], 0x00, S11, 0x698098d8); // 9 
  FF_noadd(state[3], state[0], state[1], state[2], 0x00, S12, 0x8b44f7af); // 10 
  FF_noadd(state[2], state[3], state[0], state[1], 0x00, S13, 0xffff5bb1); // 11 
  FF_noadd(state[1], state[2], state[3], state[0], 0x00, S14, 0x895cd7be); // 12 
  FF_noadd(state[0], state[1], state[2], state[3], 0x00, S11, 0x6b901122); // 13 
  FF_noadd(state[3], state[0], state[1], state[2], 0x00, S12, 0xfd987193); // 14 
  FF(state[2], state[3], state[0], state[1], 128, S13, 0xa679438e); // 15 
  FF_noadd(state[1], state[2], state[3], state[0], 0x00, S14, 0x49b40821); // 16 

  // Round 2
  GG(state[0], state[1], state[2], state[3], seed[1], S21, 0xf61e2562); // 17 
  GG_noadd(state[3], state[0], state[1], state[2], 0x00, S22, 0xc040b340); // 18 
  GG_noadd(state[2], state[3], state[0], state[1], 0x00, S23, 0x265e5a51); // 19 
  GG(state[1], state[2], state[3], state[0], seed[0], S24, 0xe9b6c7aa); // 20 
  GG_noadd(state[0], state[1], state[2], state[3], 0x00, S21, 0xd62f105d); // 21 
  GG_noadd(state[3], state[0], state[1], state[2], 0x00, S22,  0x2441453); // 22 
  GG_noadd(state[2], state[3], state[0], state[1], 0x00, S23, 0xd8a1e681); // 23 
  GG(state[1], state[2], state[3], state[0], 128, S24, 0xe7d3fbc8); // 24 
  GG_noadd(state[0], state[1], state[2], state[3], 0x00, S21, 0x21e1cde6); // 25 
  GG(state[3], state[0], state[1], state[2], 128, S22, 0xc33707d6); // 26 
  GG(state[2], state[3], state[0], state[1], seed[3], S23, 0xf4d50d87); // 27 
  GG_noadd(state[1], state[2], state[3], state[0], 0x00, S24, 0x455a14ed); // 28 
  GG_noadd(state[0], state[1], state[2], state[3], 0x00, S21, 0xa9e3e905); // 29 
  GG(state[3], state[0], state[1], state[2], seed[2], S22, 0xfcefa3f8); // 30 
  GG_noadd(state[2], state[3], state[0], state[1], 0x00, S23, 0x676f02d9); // 31 
  GG_noadd(state[1], state[2], state[3], state[0], 0x00, S24, 0x8d2a4c8a); // 32 

  // Round 3
  HH_noadd(state[0], state[1], state[2], state[3], 0x00, S31, 0xfffa3942); // 33 
  HH_noadd(state[3], state[0], state[1], state[2], 0x00, S32, 0x8771f681); // 34 
  HH_noadd(state[2], state[3], state[0], state[1], 0x00, S33, 0x6d9d6122); // 35 
  HH(state[1], state[2], state[3], state[0], 128, S34, 0xfde5380c); // 36 
  HH(state[0], state[1], state[2], state[3], seed[1], S31, 0xa4beea44); // 37 
  HH(state[3], state[0], state[1], state[2], 128, S32, 0x4bdecfa9); // 38 
  HH_noadd(state[2], state[3], state[0], state[1], 0x00, S33, 0xf6bb4b60); // 39 
  HH_noadd(state[1], state[2], state[3], state[0], 0x00, S34, 0xbebfbc70); // 40 
  HH_noadd(state[0], state[1], state[2], state[3], 0x00, S31, 0x289b7ec6); // 41 
  HH(state[3], state[0], state[1], state[2], seed[0], S32, 0xeaa127fa); // 42 
  HH(state[2], state[3], state[0], state[1], seed[3], S33, 0xd4ef3085); // 43 
  HH_noadd(state[1], state[2], state[3], state[0], 0x00, S34,  0x4881d05); // 44 
  HH_noadd(state[0], state[1], state[2], state[3], 0x00, S31, 0xd9d4d039); // 45 
  HH_noadd(state[3], state[0], state[1], state[2], 0x00, S32, 0xe6db99e5); // 46 
  HH_noadd(state[2], state[3], state[0], state[1], 0x00, S33, 0x1fa27cf8); // 47 
  HH(state[1], state[2], state[3], state[0], seed[2], S34, 0xc4ac5665); // 48 

  // Round 4
  II(state[0], state[1], state[2], state[3], seed[0], S41, 0xf4292244); // 49 
  II_noadd(state[3], state[0], state[1], state[2], 0x00, S42, 0x432aff97); // 50 
  II(state[2], state[3], state[0], state[1], 128, S43, 0xab9423a7); // 51 
  II_noadd(state[1], state[2], state[3], state[0], 0x00, S44, 0xfc93a039); // 52 
  II_noadd(state[0], state[1], state[2], state[3], 0x00, S41, 0x655b59c3); // 53 
  II(state[3], state[0], state[1], state[2], seed[3], S42, 0x8f0ccc92); // 54 
  II_noadd(state[2], state[3], state[0], state[1], 0x00, S43, 0xffeff47d); // 55 
  II(state[1], state[2], state[3], state[0], seed[1], S44, 0x85845dd1); // 56 
  II_noadd(state[0], state[1], state[2], state[3], 0x00, S41, 0x6fa87e4f); // 57 
  II_noadd(state[3], state[0], state[1], state[2], 0x00, S42, 0xfe2ce6e0); // 58 
  II_noadd(state[2], state[3], state[0], state[1], 0x00, S43, 0xa3014314); // 59 
  II_noadd(state[1], state[2], state[3], state[0], 0x00, S44, 0x4e0811a1); // 60 
  II(state[0], state[1], state[2], state[3], 128, S41, 0xf7537e82); // 61 
  II_noadd(state[3], state[0], state[1], state[2], 0x00, S42, 0xbd3af235); // 62 
  II(state[2], state[3], state[0], state[1], seed[2], S43, 0x2ad7d2bb); // 63 
  II_noadd(state[1], state[2], state[3], state[0], 0x00, S44, 0xeb86d391); // 64 
  
  state[0] += 0x67452301;
  state[1] += 0xEFCDAB89;
  state[2] += 0x98BADCFE;
  state[3] += 0x10325476;

  // Done
}

#endif // __PRNG_HPP
