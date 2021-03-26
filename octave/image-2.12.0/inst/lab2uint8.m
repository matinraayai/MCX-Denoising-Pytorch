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
## Convert L*a*b* data to uint8 precision.
##
## @var{lab} must be a L*a*b* image or colormap, i.e., its dimensions
## must be MxNx3xK or Mx3.  Its type must be double, single, uint16,
## or uint8.
##
## When converted from double or single, L* values must range from 0 to
## 100, while a* and b* range from -128 to 127.  Values outside this range
## will be capped.
##
## @seealso{lab2double, lab2rgb, lab2single, lab2uint8, lab2uin16, lab2xyz}
## @end deftypefn

function [lab] = lab2uint8 (lab)
  if (nargin () != 1)
    print_usage ();
  endif
  lab = lab2cls (lab, "uint8");
endfunction

## Instead of testing the lab2uint8 function here, we test the
## conversion from uint8 type.  The actual tests for lab2uint8,
## are spread all other lab2* functions.  This makes the tests
## simpler.

%!test
%! cm_uint8 = uint8 ([0 1 2 3 4 127 128 200 254 255]);
%! cm_uint8 = repmat (cm_uint8(:), [1 3]);
%! im2d_uint8 = reshape (cm_uint8, [5 2 3]);
%! imnd_uint8 = permute (im2d_uint8, [1 4 3 2]);
%!
%! cm_uint16 = uint16 ([0  256  512  768  1024  32512  32768 51200  65024  65280]);
%! cm_uint16 = repmat (cm_uint16(:), [1 3]);
%! assert (lab2uint16 (cm_uint8), cm_uint16)
%! im2d_uint16 = reshape (cm_uint16, [5 2 3]);
%! assert (lab2uint16 (im2d_uint8), im2d_uint16)
%! assert (lab2uint16 (imnd_uint8), permute (im2d_uint16, [1 4 3 2]))
%!
%! l1 = 100/255;
%! cm = [
%!       0  -128  -128
%!      l1  -127  -127
%!    2*l1  -126  -126
%!    3*l1  -125  -125
%!    4*l1  -124  -124
%!  127*l1    -1    -1
%!  128*l1     0     0
%!  200*l1    72    72
%!  254*l1   126   126
%!     100   127   127];
%! im2d = reshape (cm, [5 2 3]);
%! imnd = permute (im2d, [1 4 3 2]);
%!
%! assert (lab2double (cm_uint8), cm)
%! assert (lab2double (im2d_uint8), im2d)
%! assert (lab2double (imnd_uint8), imnd)
%!
%! assert (lab2single (cm_uint8), single (cm))
%! assert (lab2single (im2d_uint8), single (im2d))
%! assert (lab2single (imnd_uint8), single (imnd))
