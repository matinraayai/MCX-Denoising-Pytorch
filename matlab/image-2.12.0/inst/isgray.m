## Copyright (C) 2000 Kai Habel <kai.habel@gmx.de>
## Copyright (C) 2011, 2015 Carnë Draug <carandraug+dev@gmail.com>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} isgray (@var{img})
## Return true if @var{img} is a grayscale image.
##
## A variable can be considered a grayscale image if it is a non-sparse,
## real array of size @nospell{MxNx1xK}, and:
##
## @itemize @bullet
## @item of floating point class with values in the [0 1] range or NaN;
## @item of class uint8, uint16, or int16.
## @end itemize
##
## @emph{Note}: despite their suggestive names, the functions isbw,
## isgray, isind, and isrgb, are ambiguous since it is not always possible
## to distinguish between those image types.  For example: an uint8 matrix
## can be both a grayscale and indexed image; a grayscale image may have
## values outside the range [0 1].  They are good to dismiss input as an
## invalid image type, but not for identification.
##
## @seealso{gray2ind, isbw, isind, isrgb}
## @end deftypefn

function bool = isgray (img)
  if (nargin () != 1)
    print_usage ();
  endif

  bool = false;
  if (isimage (img) && ndims (img) < 5 && size (img, 3) == 1)
    if (isfloat (img))
      bool = ispart (@is_float_image, img);
    elseif (any (isa (img, {"uint8", "uint16", "int16"})))
      bool = true;
    endif
  endif
endfunction

%!assert (isgray ([0 0 1; 1 0 1]), true)
%!assert (isgray (zeros (3)), true)
%!assert (isgray (ones (3)), true)

%!test
%! a = rand (10);
%! assert (isgray (a), true);
%! a(5, 5) = 2;
%! assert (isgray (a), false);

%!test
%! a = uint8 (randi (255, 10));
%! assert (isgray (a), true);
%! a = int8 (a);
%! assert (isgray (a), false);

%!test
%! a = rand (10);
%! a(50) = NaN;
%! assert (isgray (a), true);

%!assert (isgray (rand (5, 5, 1, 4)), true);
%!assert (isgray (rand (5, 5, 3, 4)), false);
%!assert (isgray (rand (5, 5, 3)), false);
%!assert (isgray (rand (5, 5, 1, 3, 4)), false);

%!assert (isgray (rand (5, "single")), true)

## While having some NaN is ok, having all NaN is not
%!assert (isgray ([.1 .2 .3; .4 NaN .6; .7 .8 .9]), true)
%!assert (isgray ([.1 .2 .3; NA NaN .6; .7 .8 .9]), true)
%!assert (isgray ([.1 .2 .3; NA  .5 .6; .7 .8 .9]), true)
%!assert (isgray (NaN (5)), false)
%!assert (isgray (NA (5)), false)
