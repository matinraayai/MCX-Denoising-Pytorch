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
## @deftypefn {} {[@var{xIntrinsic}, @var{yIntrinsic}, @var{zIntrinsic}] =} worldToIntrinsic (@var{r}, @var{xWorld}, @var{yWorld}, @var{zWorld})
## Convert from world to intrinsic coordinates.
##
## Converts world coordinates @var{xWorld}, @var{yWorld} and @var{zWorld} to
## intrinsic coordinates @var{xIntrinsic}, @var{yIntrinsic} and @var{zIntrinsic}
## of an image associated with the spatial referencing object @var{r}.  If a
## point (@var{xWorld}(i), @var{yWorld}(i), @var{zWorld}(i)) falls outside
## the bounds of the image, its intrinsic coordinates are extrapolated,
## possibly resulting in negative values.
##
## @seealso{imref2d, imref3d, intrinsicToWorld}
## @end deftypefn

function [xIntrinsic, yIntrinsic, zIntrinsic] = worldToIntrinsic (r, xWorld, yWorld, zWorld)
  if (nargin != 4)
    print_usage ();
  endif
  
  validateattributes (xWorld, {"numeric"}, ...
  {"real"}, "imref3d", "xWorld");
  validateattributes (yWorld, {"numeric"}, ...
  {"real"}, "imref3d", "yWorld");
  validateattributes (zWorld, {"numeric"}, ...
  {"real"}, "imref3d", "zWorld");
  
  if (! all (size (xWorld) == size (yWorld)) ...
    || ! all (size (xWorld) == size (zWorld)))
    error ("Octave:invalid-input-arg", ...
    "xWorld, yWorld and zWorld must be of the same size");
  endif
  
  xIntrinsicLimits = r.XIntrinsicLimits;
  yIntrinsicLimits = r.YIntrinsicLimits;
  zIntrinsicLimits = r.ZIntrinsicLimits;
  xWorldLimits = r.XWorldLimits;
  yWorldLimits = r.YWorldLimits;
  zWorldLimits = r.ZWorldLimits;
  xIntrinsic = xIntrinsicLimits(1) + (xWorld  - xWorldLimits(1)) ...
  / r.PixelExtentInWorldX;
  yIntrinsic = yIntrinsicLimits(1) + (yWorld  - yWorldLimits(1)) ...
  / r.PixelExtentInWorldY;
  zIntrinsic = zIntrinsicLimits(1) + (zWorld  - zWorldLimits(1)) ...
  / r.PixelExtentInWorldZ;
endfunction

%!error id=Octave:invalid-fun-call worldToIntrinsic (imref3d)
%!error id=Octave:invalid-fun-call worldToIntrinsic (imref3d, 1, 2)
%!error id=Octave:invalid-fun-call worldToIntrinsic (imref3d, 1, 2, 3, 4)
%!error id=Octave:expected-real worldToIntrinsic (imref3d, 1j, 2, 3)
%!error id=Octave:expected-real worldToIntrinsic (imref3d, 1, 2j, 3)
%!error id=Octave:expected-real worldToIntrinsic (imref3d, 1, 2, 3j)
%!error id=Octave:invalid-input-arg worldToIntrinsic (imref3d, [1, 2], 3, 4)
%!error id=Octave:invalid-input-arg worldToIntrinsic (imref3d, 1, [2, 3], 4)
%!error id=Octave:invalid-input-arg worldToIntrinsic (imref3d, 1, 2, [3, 4])

%!test
%! r = imref3d ([128, 128, 27], 2, 2, 4);
%! xW = [108, 108, 108.2, 2];
%! yW = [92, 92, 92, -1];
%! zW = [52, 55, 52, 0.33];
%! [xI, yI, zI] = worldToIntrinsic (r, xW, yW, zW);
%! assert (xI, [54, 54, 54.1, 1], 1e-6)
%! assert (yI, [46, 46, 46, -0.5], 1e-6)
%! assert (zI, [13, 13.75, 13, 0.0825], 1e-6)