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
## Convert L*a*b* data to uint16 precision.
##
## @var{lab} must be a L*a*b* image or colormap, i.e., its dimensions
## must be MxNx3xK or Mx3.  Its type must be double, single, uint16,
## or uint8.
##
## When converted from double or single, L* values usually range from 0 to
## 100, while a* and b* range from -128 to 127.  The upper limit values
## will be converted to 65280.
##
## @seealso{lab2double, lab2rgb, lab2single, lab2uint8, lab2uin16, lab2xyz}
## @end deftypefn

function [lab] = lab2uint16 (lab)
  if (nargin () != 1)
    print_usage ();
  endif
  lab = lab2cls (lab, "uint16");
endfunction

## Instead of testing the lab2uint16 function here, we test the
## conversion from uint16 type.  The actual tests for lab2uint16,
## are spread all other lab2* functions.  This makes the tests
## simpler.

%!test
%! cm_uint16 = uint16 ([0 127 128 383 384 65151 65152 65279 65280 65281 65534 65535]);
%! cm_uint16 = repmat (cm_uint16(:), [1 3]);
%! im2d_uint16 = reshape (cm_uint16, [4 3 3]);
%! imnd_uint16 = permute (im2d_uint16, [1 4 3 2]);
%!
%! cm_uint8 = uint8 ([0 0 1 1 2 254 255 255 255 255 255 255]);
%! cm_uint8 = repmat (cm_uint8(:), [1 3]);
%! assert (lab2uint8 (cm_uint16), cm_uint8)
%! im2d_uint8 = reshape (cm_uint8, [4 3 3]);
%! assert (lab2uint8 (im2d_uint16), im2d_uint8)
%! assert (lab2uint8 (imnd_uint16), permute (im2d_uint8, [1 4 3 2]))
%!
%! l1 = 100/65280;
%! ab1 = 255/65280;
%! cm = [
%!         0  -128
%!    127*l1  -128+(ab1*127)
%!    128*l1  -128+(ab1*128)
%!    383*l1  -128+(ab1*383)
%!    384*l1  -128+(ab1*384)
%!  65151*l1  -128+(ab1*65151)
%!  65152*l1  -128+(ab1*65152)
%!  65279*l1  -128+(ab1*65279)
%!       100   127
%!  65281*l1  -128+(ab1*65281)
%!  65534*l1  -128+(ab1*65534)
%!  65535*l1  -128+(ab1*65535)];
%! cm(:,3) = cm(:,2);
%! im2d = reshape (cm, [4 3 3]);
%! imnd = permute (im2d, [1 4 3 2]);
%!
%! assert (lab2double (cm_uint16), cm)
%! assert (lab2double (im2d_uint16), im2d)
%! assert (lab2double (imnd_uint16), imnd)
%!
%! assert (lab2single (cm_uint16), single (cm))
%! assert (lab2single (im2d_uint16), single (im2d))
%! assert (lab2single (imnd_uint16), single (imnd))
