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
## @deftypefn {Function File} {} im2int16 (@var{img})
## Convert image to int16.
##
## The conversion of @var{img} to a 16-bit signed integer, is dependent
## on the type of input image.  The following input classes are supported:
##
## @table @samp
## @item uint8 or uint16
## Values are rescaled to the range of the int16 class [-32768 32767].
##
## @item logical
## True and false values are assigned a value of 32767 and -32768 respectively.
##
## @item double or single
## Values are truncated to the interval [0 1] and then rescaled to the range
## of values of the int16 class [-32768 32767].
##
## @item int16
## Returns the same image.
##
## @end table
##
## @seealso{im2bw, imcast, im2uint8, im2double, im2single, im2uint16}
## @end deftypefn

function imout = im2int16 (img)
  if (nargin != 1)
    print_usage ();
  endif
  imout = imcast (img, "int16");
endfunction

%!assert (im2int16 (int16 ([-2 2 3])),    int16 ([-2 2 3]));
%!assert (im2int16 (uint16 ([0 65535])),  int16 ([-32768 32767]));
%!assert (im2int16 ([false true]),        int16 ([-32768 32767]));
%!assert (im2int16 ([true false]),        int16 ([32767 -32768]));
%!assert (im2int16 (uint8 ([0 127 128 255])), int16 ([-32768 -129 128 32767]));

%!assert (im2int16 ([0 1.4/65535 1.5/65535 2/65535 1]), int16 ([-32768 -32767 -32766 -32766 32767]));

%!assert (im2int16 ([0 0.5 1]),  int16 ([-32768 0 32767]));
%!assert (im2int16 ([-1 0 1 2]), int16 ([-32768 -32768 32767 32767]));

%!error im2int16 ([1 2], "indexed");

