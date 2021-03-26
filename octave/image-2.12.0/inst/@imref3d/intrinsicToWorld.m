## Copyright (C) 2018 Martin Janda <janda.martin1@gmail.com>
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {[@var{xWorld}, @var{yWorld}, @var{zWorld}] =} intrinsicToWorld (@var{r}, @var{xIntrinsic}, @var{yIntrinsic}, @var{zIntrinsic})
## Convert from intrinsic to world coordinates.
##
## Converts intrinsic coordinates of the image associated with the spatial
## referencing object @var{r} to world coordinates @var{xWorld}, @var{yWorld}
## and @var{zWorld}.  If a point
## (@var{xIntrinsic}(i), @var{yIntrinsic}(i), @var{zIntrinsic}(i))
## falls outside the intrinsic bounds of the image, the world coordinates are
## extrapolated, possibly resulting in negative values.
##
## @seealso{imref2d, imref3d, worldToIntrinsic}
## @end deftypefn

function [xWorld, yWorld, zWorld] = intrinsicToWorld (r, xIntrinsic, yIntrinsic, zIntrinsic)
  if (nargin != 4)
    print_usage ();
  endif
  
  validateattributes (xIntrinsic, {"numeric"}, ...
  {"real"}, "imref3d", "xIntrinsic");
  validateattributes (yIntrinsic, {"numeric"}, ...
  {"real"}, "imref3d", "yIntrinsic");
  validateattributes (zIntrinsic, {"numeric"}, ...
  {"real"}, "imref3d", "zIntrinsic");
  
  if (! all (size (xIntrinsic) == size (yIntrinsic)) ...
    || ! all (size (xIntrinsic) == size (zIntrinsic)))
    error ("Octave:invalid-input-arg", ...
    "xIntrinsic, yIntrinsic and zIntrinsic must be of the same size");
  endif
  
  xIntrinsicLimits = r.XIntrinsicLimits;
  yIntrinsicLimits = r.YIntrinsicLimits;
  zIntrinsicLimits = r.ZIntrinsicLimits;
  xWorldLimits = r.XWorldLimits;
  yWorldLimits = r.YWorldLimits;
  zWorldLimits = r.ZWorldLimits;
  xWorld = xWorldLimits(1) + r.PixelExtentInWorldX * ...
  (xIntrinsic - xIntrinsicLimits(1));
  yWorld = yWorldLimits(1) + r.PixelExtentInWorldY * ...
  (yIntrinsic - yIntrinsicLimits(1));
  zWorld = zWorldLimits(1) + r.PixelExtentInWorldZ * ...
  (zIntrinsic - zIntrinsicLimits(1));
endfunction

%!error id=Octave:invalid-fun-call intrinsicToWorld (imref3d)
%!error id=Octave:invalid-fun-call intrinsicToWorld (imref3d, 1)
%!error id=Octave:invalid-fun-call intrinsicToWorld (imref3d, 1, 2)
%!error id=Octave:invalid-fun-call intrinsicToWorld (imref3d, 1, 2, 3, 4)
%!error id=Octave:expected-real intrinsicToWorld (imref3d, 1j, 2, 3)
%!error id=Octave:expected-real intrinsicToWorld (imref3d, 1, 2j, 3)
%!error id=Octave:expected-real intrinsicToWorld (imref3d, 1, j, 3j)
%!error id=Octave:invalid-input-arg intrinsicToWorld (imref3d, [1, 2], 3, 4)
%!error id=Octave:invalid-input-arg intrinsicToWorld (imref3d, 1, [2, 3], 4)
%!error id=Octave:invalid-input-arg intrinsicToWorld (imref3d, 1, 2, [3, 4])

%!test
%! r = imref3d ([128, 128, 27], 2, 2, 4);
%! xI = [54, 71, 57, 70];
%! yI = [46, 48, 79, 80];
%! zI = [13, 13, 13, 13];
%! [xW, yW, zW] = intrinsicToWorld (r, xI, yI, zI);
%! assert (xW, [108, 142, 114, 140])
%! assert (yW, [92, 96, 158, 160])
%! assert (zW, [52, 52, 52, 52])

%!test
%! [xW, yW, zW] = intrinsicToWorld (imref3d, -5.3, -2.8, -15.88);
%! assert (xW, -5.3)
%! assert (yW, -2.8)
%! assert (zW, -15.88, 1e-6)

%!test
%! [xW, yW, zW] = intrinsicToWorld (imref3d, [1, 2; 3, 4],
%!                                           [2, 3; 5, 9],
%!                                           [-5, 8; 19, 42.8]);
%! assert (xW, [1, 2; 3, 4])
%! assert (yW, [2, 3; 5, 9])
%! assert (zW, [-5, 8; 19, 42.8])
