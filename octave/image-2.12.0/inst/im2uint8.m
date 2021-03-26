## Copyright (C) 2007 Søren Hauberg <soren@hauberg.org>
## Copyright (C) 2012-2014 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn  {Function File} {} im2uint8 (@var{img})
## @deftypefnx {Function File} {} im2uint8 (@var{img}, "indexed")
## Convert image to uint8.
##
## The conversion of @var{img} to an 8-bit unsigned integer, is dependent
## on the type of input image.  The following input classes are supported
## for non-indexed images:
##
## @table @samp
## @item int16 or uint16
## Values are rescaled to the range of the uint8 class [0 255].
##
## @item logical
## True and false values are assigned a value of 0 and 255 respectively.
##
## @item double or single
## Values are truncated to the interval [0 1] and then rescaled to the range
## of values of the uint8 class [0 255].
##
## @item uint8
## Returns the same image.
##
## @end table
##
## If the second argument is the string @qcode{"indexed"}, then values are
## cast to uint8, and a -1 offset is applied if input is
## a floating point class.  Input checking is performed and an error will
## be throw is the range of values in uint8 is not enough for all the
## image indices.
##
## @seealso{im2bw, imcast, im2double, im2int16, im2single, im2uint16}
## @end deftypefn

function imout = im2uint8 (img, varargin)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (nargin == 2 && ! strcmpi (varargin{1}, "indexed"))
    error ("im2uint8: second input argument must be the string \"indexed\"");
  endif
  imout = imcast (img, "uint8", varargin{:});
endfunction

%!assert (im2uint8 (uint8 ([1 2 3])), uint8 ([1 2 3]));
%!assert (im2uint8 (uint16 ([0 65535])), uint8 ([0 255]));
%!assert (im2uint8 ([0 0.5 1]), uint8 ([0 128 255]));
%!assert (im2uint8 ([1 2]), uint8 ([255 255]));
%!assert (im2uint8 ([-1 0 0.5 1 2]), uint8 ([0 0 128 255 255]));
%!assert (im2uint8 (int16 ([-32768 0 32768])), uint8 ([0 128 255]));
%!assert (im2uint8 ([false true]), uint8 ([0 255]));
%!assert (im2uint8 ([true false]), uint8 ([255 0]));

%!assert (im2uint8 ([1 256], "indexed"), uint8 ([0 255]));
%!assert (im2uint8 ([3 25], "indexed"), uint8 ([2 24]));
%!assert (im2uint8 (uint16 ([3 25]), "indexed"), uint8 ([3 25]));

%!error <indexed> im2uint8 ([0 1 2], "indexed");
%!error <indexed> im2uint8 (int16 ([17 8]), "indexed");
%!error <indexed> im2uint8 (int16 ([-7 8]), "indexed");
%!error <indexed> im2uint8 ([false true], "indexed");
%!error <range of values> im2uint8 (uint16 (256), "indexed");
%!error <range of values> im2uint8 (257, "indexed");

%!assert (im2uint8 ((1:255) ./ 256), uint8 ([1:128 128:254]))
%!assert (im2uint8 ((0:255) ./ 256), uint8 ([0:128 128:254]))
