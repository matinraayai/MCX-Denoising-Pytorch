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
## @deftypefn  {Function File} {} im2uint16 (@var{img})
## @deftypefnx {Function File} {} im2uint16 (@var{img}, "indexed")
## Convert image to uint16.
##
## The conversion of @var{img} to a 16-bit unsigned integer, is dependent
## on the type of input image.  The following input classes are supported
## for non-indexed images:
##
## @table @samp
## @item int16 or uint8
## Values are rescaled to the range of the uint16 class [0 65535].
##
## @item logical
## True and false values are assigned a value of 0 and 255 respectively.
##
## @item double or single
## Values are truncated to the interval [0 1] and then rescaled to the range
## of values of the int16 class [0 255].
##
## @item uint16
## Returns the same image.
##
## @end table
##
## If the second argument is the string @qcode{"indexed"}, then values are
## cast to uint16, and a -1 offset is applied if input is
## a floating point class.  Input checking is performed and an error will
## be throw is the range of values in uint16 is not enough for all the
## image indices.
##
## @seealso{im2bw, imcast, im2uint8, im2double, im2int16, im2single}
## @end deftypefn

function imout = im2uint16 (im, varargin)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (nargin == 2 && ! strcmpi (varargin{1}, "indexed"))
    error ("im2uint16: second input argument must be the string \"indexed\"");
  endif
  imout = imcast (im, "uint16", varargin{:});
endfunction

%!assert (im2uint16 (uint16 ([1 2 3])), uint16 ([1 2 3]));
%!assert (im2uint16 (uint8 ([0 127 128 255])), uint16 ([0 32639 32896 65535]));
%!assert (im2uint16 ([0 0.5 1]), uint16 ([0 32768 65535]));
%!assert (im2uint16 ([0 1/65535 1.4/65535 1.5/65535 1]), uint16 ([0 1 1 2 65535]));
%!assert (im2uint16 ([1 2]), uint16 ([65535 65535]));
%!assert (im2uint16 ([-1 0 0.5 1]), uint16 ([0 0 32768 65535]));
%!assert (im2uint16 (int16 ([-32768 -1 0 32768])), uint16 ([0 32767 32768 65535]));
%!assert (im2uint16 ([false true]), uint16 ([0 65535]));
%!assert (im2uint16 ([true false]), uint16 ([65535 0]));

%!assert (im2uint16 (uint8 ([3 25]), "indexed"), uint16 ([3 25]));
%!assert (im2uint16 ([1 3 25], "indexed"), uint16 ([0 2 24]));

%!error <indexed> im2uint16 ([0 1 2], "indexed");
%!error <indexed> im2uint16 (int16 ([17 8]), "indexed");
%!error <indexed> im2uint16 (int16 ([-7 8]), "indexed");
%!error <indexed> im2uint16 ([false true], "indexed");
%!error <range of values> im2uint16 (65537, "indexed");

