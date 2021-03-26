// Copyright (C) 2013-2015 Carnë Draug <carandraug@octave.org>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see
// <http://www.gnu.org/licenses/>.

// This function is implemented in C++ because it basically does
// indexing with base 0.  If implemented in a m file, it would
// require conversion of the image to a float just to add 1.

#include <string>

#include <octave/dim-vector.h>
#include <octave/oct-inttypes.h>

#include <octave/defun-dld.h>
#include <octave/defun-int.h>
#include <octave/error.h>
#include <octave/ovl.h>

template<class P>
static inline P
intlut_index (const typename P::val_type A, const P lut_vec[])
{
  return lut_vec[A];
}

template<>
inline octave_int16
intlut_index (const typename octave_int16::val_type A,
              const octave_int16 lut_vec[])
{
  return lut_vec[32768 + A];
}

template<class T>
static T
intlut (const T& A, const T& lut)
{
  const auto* A_vec = A.fortran_vec ();
  const auto* lut_vec = lut.fortran_vec ();

  T B (A.dims ());
  auto* B_vec = B.fortran_vec ();

  const octave_idx_type n = A.numel ();

  typedef typename T::element_type::val_type P_val_type;
  for (octave_idx_type i = 0; i < n; i++, B_vec++, A_vec++)
    *B_vec = intlut_index (static_cast<P_val_type> (*A_vec), lut_vec);

  return B;
}


DEFUN_DLD (intlut, args, , "\
-*- texinfo -*-\n\
@deftypefn {Function File} {} intlut (@var{A}, @var{LUT})\n\
Convert integer values with lookup table (LUT).\n\
\n\
Replace the values from the array @var{A} with the corresponding\n\
value from the lookup table @var{LUT}.  This is equivalent as indexing\n\
@var{LUT} with @var{A}, with a base equal to @var{A} minimum possible\n\
value, i.e., @code{intmin (@var{A})}.\n\
\n\
For the simplest case of uint8 and uint16 class, it corresponds to:\n\
\n\
@example\n\
@var{LUT}(double (@var{A}) +1)\n\
@end example\n\
\n\
but without the temporary conversion of @var{A} to floating point\n\
thus reducing memory usage.\n\
\n\
@var{A} and @var{LUT} must be of the same class, and uint8, uint16,\n\
or int16.  @var{LUT} must have exactly 256 elements for class uint8,\n\
and 65536 for classes uint16 and int16.  Output is of same class\n\
as @var{LUT}.\n\
\n\
@seealso{ind2gray, ind2rgb, rgb2ind}\n\
@end deftypefn")
{
  octave_value_list rv (1);

  if (args.length () != 2)
    print_usage ();

  const std::string cls = args(0).class_name ();
  if (cls != args(1).class_name ())
    error ("intlut: A and LUT must be of same class");

  const dim_vector lut_dims = args(1).dims ();
  if (lut_dims.length () != 2 || (lut_dims(0) > 1 && lut_dims(1) > 1))
    error ("intlut: LUT must be a vector");

#define IF_TYPE(TYPE, TYPE_RANGE) \
  if (args(0).is_ ## TYPE ## _type ()) \
    { \
      if (args(1).numel () != TYPE_RANGE) \
        error ("intlut: LUT must have " #TYPE_RANGE " elements for class %s", \
               cls.c_str ()); \
      rv(0) = intlut (args(0).TYPE ## _array_value (), \
                      args(1).TYPE ## _array_value ()); \
    }

  IF_TYPE(uint8, 256)
  else IF_TYPE(uint16, 65536)
  else IF_TYPE(int16, 65536)
  else
    error ("intlut: A must be of class uint8, uint16, or int16");

#undef IF_TYPE

  return rv;
}

/*
%!assert (intlut (uint8  (1:4), uint8  (  255:-1:0)), uint8  (254:-1:251));
%!assert (intlut (uint16 (1:4), uint16 (65535:-1:0)), uint16 (65534:-1:65531));
%!assert (intlut (int16  (1:4), int16  (32767:-1:-32768)), int16 (-2:-1:-5));

%!assert (intlut (uint8 (255), uint8 (0:255)), uint8 (255));
%!assert (intlut (uint16 (65535), uint16 (0:65535)), uint16 (65535));
%!assert (intlut (int16 (32767), int16 (-32768:32767)), int16 (32767));

%!error intlut ()
%!error intlut ("text")
%!error <must be of same class> intlut (1:20, uint8 (0:255));
%!error <must be of same class> intlut (uint16 (1:20), uint8 (0:255));
%!error <must have 256 elements> intlut (uint8 (1:20), uint8 (0:200));
%!error <must have 65536 elements> intlut (uint16 (1:20), uint16 (0:500));
%!error <LUT must be a vector> intlut (uint8 (56), uint8 (magic (16) -1))
*/
