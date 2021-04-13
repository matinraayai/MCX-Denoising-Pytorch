## Copyright (C) 2000 Kai Habel <kai.habel@gmx.de>
## Copyright (C) 2004 Josep Monés i Teixidor <jmones@puntbarra.com>
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
## @deftypefn {Function File} {} isrgb (@var{img})
## Return true if @var{img} is a RGB image.
##
## A variable can be considered a RGB image if it is a non-sparse,
## real array of size @nospell{MxNx3xK}, and:
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
## @seealso{rgb2gray, rgb2ind, isbw, isgray, isind}
## @end deftypefn

function bool = isrgb (img)

  if (nargin != 1)
    print_usage;
  endif

  bool = false;
  if (isimage (img) && ndims (img) < 5 && size (img, 3) == 3)
    if (isfloat (img))
      bool = ispart (@is_float_image, img);
    elseif (any (isa (img, {"uint8", "uint16", "int16"})))
      bool = true;
    endif
  endif

endfunction

## Non-matrix
%!assert (isrgb ("this is not a RGB image"), false);

## Double matrix tests
%!assert (isrgb (rand (5, 5)), false);
%!assert (isrgb (rand (5, 5, 1, 5)), false);
%!assert (isrgb (rand (5, 5, 3, 5)), true);
%!assert (isrgb (rand (5, 5, 3)), true);
%!assert (isrgb (ones (5, 5, 3)), true);
%!assert (isrgb (ones (5, 5, 3) + eps), false);
%!assert (isrgb (zeros (5, 5, 3) - eps), false);

%!assert (isrgb (rand (5, 5, 3) > 0.5), false);
%!assert (isrgb (randi ([-100 100], 5, 5, 3, "int16")), true)
