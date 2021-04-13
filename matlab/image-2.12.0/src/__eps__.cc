/*
Copyright (C) 1994-2017 John W. Eaton
Copyright (C) 2009 Jaroslav Hajek
Copyright (C) 2009-2010 VZLU Prague
Copyright (C) 2012 Carlo de Falco

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, see
<http://www.gnu.org/licenses/>.
*/

#include <cmath>
#include <limits>

#include <octave/oct.h>
#include <octave/lo-mappers.h>

// Work around bug #50561 - remove this when the Image package is
// dependent on Octave 4.4 or later only.
//
// When computing the eps values for an array, the eps function in
// core is unnaceptably slow in versions 4.2 and older.  Just try
// something like `eps (rand (1024, 1024))`.  This issue was fixed
// sometime during the 4.3 development versions.
//
// This is the fixed implementation and to be used by the image
// package.  It does no input check and only works with an array
// input.

template <typename T>
T
eps (const T& x)
{
  T epsval = x.abs ();
  typedef typename T::value_type P;
  for (octave_idx_type i = 0; i < x.numel (); i++)
    {
      P val = epsval.xelem (i);
      if (octave::math::isnan (val) || octave::math::isinf (val))
        epsval(i) = octave::numeric_limits<P>::NaN ();
      else if (val < std::numeric_limits<P>::min ())
        epsval(i) = std::numeric_limits<P>::denorm_min ();
      else
        {
          int exponent;
          octave::math::frexp (val, &exponent);
          const P digits = std::numeric_limits<P>::digits;
          epsval(i) = std::pow (static_cast<P> (2.0),
                                static_cast<P> (exponent - digits));
        }
    }
  return epsval;
}

DEFUN_DLD (__eps__, args, , "")
{
  octave_value retval;
  octave_value arg0 = args(0);
  if (arg0.is_single_type ())
    {
      FloatNDArray epsval = eps (arg0.float_array_value ());
      retval = epsval;
    }
  else
    {
      NDArray epsval = eps (arg0.array_value ());
      retval = epsval;
    }
  return retval;
}
