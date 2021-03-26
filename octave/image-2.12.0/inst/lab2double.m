## Copyright (C) 2016 Carnë Draug <carandraug@octave.org>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} lab2double (@var{lab})
## Convert L*a*b* data to double precision.
##
## @var{lab} must be a L*a*b* image or colormap, i.e., its dimensions
## must be MxNx3xK or Mx3.  Its type must be double, single, uint16,
## or uint8.
##
## When converted to double, L* values range from 0 to 100, while a* and
## b* range from -128 to 127.  When converting from uint16, the upper limit
## is 65280 (higher values will be converted above the range).
##
## @seealso{lab2double, lab2rgb, lab2single, lab2uint8, lab2uin16, lab2xyz}
## @end deftypefn

function [lab] = lab2double (lab)
  if (nargin () != 1)
    print_usage ();
  endif
  lab = lab2cls (lab, "double");
endfunction

## Instead of testing the lab2double function here, we test the
## conversion from double type.  The actual tests for lab2double,
## are spread all other lab2* functions.  This makes the tests
## simpler.

%!test
%! l_max_f = 100 + (25500 / 65280);
%! ab_max_f = 127 + (255 / 256);
%! cm = [
%!   -Inf
%!   Inf
%!   NaN
%!   l_max_f
%!   ab_max_f
%!   -200
%!   -129
%!   -128
%!   -128+(255/65280)*(0.499999)
%!   -128+(255/65280)*(0.500001) # should be 0.5, but float rounding error
%!   -128+(255/65280)*(0.500002)
%!   -127
%!   -1
%!   0
%!   (100/65280)*(0.499999)
%!   (100/65280)*(0.51)
%!   (100/65280)*(0.500001)
%!   1
%!   99
%!   100
%!   101
%!   126
%!   127
%!   128
%!   254
%!   255
%!   256
%!   257];
%! cm = repmat (cm, [1 3]);
%! im2d = reshape (cm, [7 4 3]);
%! imnd = permute (im2d, [1 4 3 2]);
%!
%! cm_uint8 = uint8 ([
%!     0    0    0
%!   255  255  255
%!   255  255  255
%!   255  228  228
%!   255  255  255
%!     0    0    0
%!     0    0    0
%!     0    0    0
%!     0    0    0
%!     0    0    0
%!     0    0    0
%!     0    1    1
%!     0  127  127
%!     0  128  128
%!     0  128  128
%!     0  128  128
%!     0  128  128
%!     3  129  129
%!   252  227  227
%!   255  228  228
%!   255  229  229
%!   255  254  254
%!   255  255  255
%!   255  255  255
%!   255  255  255
%!   255  255  255
%!   255  255  255
%!   255  255  255]);
%!
%! assert (lab2uint8 (cm), cm_uint8)
%! im2d_uint8 = reshape (cm_uint8, [7 4 3]);
%! assert (lab2uint8 (im2d), im2d_uint8)
%! assert (lab2uint8 (imnd), permute (im2d_uint8, [1 4 3 2]))
%!
%! cm_uint16 = uint16 ([
%!       0      0      0
%!   65535  65535  65535
%!   65535  65535  65535
%!   65535  58468  58468
%!   65535  65535  65535
%!       0      0      0
%!       0      0      0
%!       0      0      0
%!       0      0      0
%!       0      1      1
%!       0      1      1
%!       0    256    256
%!       0  32512  32512
%!       0  32768  32768
%!       0  32768  32768
%!       1  32768  32768
%!       1  32768  32768
%!     653  33024  33024
%!   64627  58112  58112
%!   65280  58368  58368
%!   65535  58624  58624
%!   65535  65024  65024
%!   65535  65280  65280
%!   65535  65535  65535
%!   65535  65535  65535
%!   65535  65535  65535
%!   65535  65535  65535
%!   65535  65535  65535]);
%!
%! assert (lab2uint16 (cm), cm_uint16)
%! im2d_uint16 = reshape (cm_uint16, [7 4 3]);
%! assert (lab2uint16 (im2d), im2d_uint16)
%! assert (lab2uint16 (imnd), permute (im2d_uint16, [1 4 3 2]))
%!
%! assert (lab2single (cm), single (cm))
%! assert (lab2single (im2d), single (im2d))
%! assert (lab2single (imnd), single (imnd))
