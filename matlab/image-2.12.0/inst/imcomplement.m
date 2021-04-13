## Copyright (C) 2008 Søren Hauberg <soren@hauberg.org>
## Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} imcomplement (@var{A})
## Compute image complement or negative.
##
## Intuitively this corresponds to the intensity of bright and dark
## regions being reversed.  The exact operation performed is dependent
## on the class of the image.
##
## @table @asis
## @item floating point
## Since floating point images are meant to have values in the range [0 1],
## this is equivalent @code{A -1}.  This leads to values within the range
## to be inverted while others to stay equally distant from the limits.
##
## @item logical
## Equivalent to @code{! A}
##
## @item integer
## Inverts the values within the range of the data type.  This is
## equivalent to @code{bitcmp (@var{A})}.
##
## @end table
##
## @seealso{imadd, imdivide, imlincomb, immultiply, imsubtract}
## @end deftypefn

function B = imcomplement (A)

  if (nargin != 1)
    print_usage ();
  endif

  if (isfloat (A))
    B = 1 - A;
  elseif (islogical (A))
    B = ! A;
  elseif (isinteger (A))
    if (intmin (class (A)) < 0)
      ## for signed integers, we use bitcmp
      B = bitcmp (A);
    else
      ## this is currently more efficient than bitcmp (but hopefully core
      ## will change)
      B = intmax (class (A)) - A;
    endif
  else
    error("imcomplement: A must be an image but is of class `%s'", class (A));
  endif

endfunction

%!assert (imcomplement (10), -9);
%!assert (imcomplement (single (10)), single (-9));
%!assert (imcomplement (0.2), 0.8);
%!assert (imcomplement (uint8 (0)), uint8 (255));
%!assert (imcomplement (uint8 (1)), uint8 (254));
%!assert (imcomplement (uint16 (0)), uint16 (65535));
%!assert (imcomplement (uint16 (1)), uint16 (65534));

%!assert (imcomplement (int8 (-128)), int8 ( 127));
%!assert (imcomplement (int8 ( 127)), int8 (-128));
%!assert (imcomplement (int16 (-1)), int16 ( 0));
%!assert (imcomplement (int16 ( 0)), int16 (-1));
%!assert (imcomplement (int16 ( 1)), int16 (-2));

%!assert (imcomplement ([true false true]), [false true false])

%!error <must be an image> imcomplement ("not an image")
