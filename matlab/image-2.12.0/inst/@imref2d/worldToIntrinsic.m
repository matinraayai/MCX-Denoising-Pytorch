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
## @deftypefn {} {[@var{xIntrinsic}, @var{yIntrinsic}] =} worldToIntrinsic (@var{r}, @var{xWorld}, @var{yWorld})
## Convert from world to intrinsic coordinates.
##
## Converts world coordinates @var{xWorld} and @var{yWorld} to intrinsic
## coordinates @var{xIntrinsic} and @var{yIntrinsic} of an image associated
## with the spatial referencing object @var{r}.  If a point
## (@var{xWorld}(i), @var{yWorld}(i)) falls outside the bounds of the image,
## its intrinsic coordinates are extrapolated, possibly resulting in
## negative values.
##
## @seealso{imref2d, imref3d, intrinsicToWorld}
## @end deftypefn

function [xIntrinsic, yIntrinsic] = worldToIntrinsic (r, xWorld, yWorld)
  if (nargin != 3)
    print_usage ();
  endif
  
  validateattributes (xWorld, {"numeric"}, ...
  {"real"}, "imref2d", "xWorld");
  validateattributes (yWorld, {"numeric"}, ...
  {"real"}, "imref2d", "yWorld");
  
  if (! all (size (xWorld) == size (yWorld)))
    error ("Octave:invalid-input-arg", ...
    "xWorld and yWorld must be of the same size");
  endif
  
  xIntrinsicLimits = r.XIntrinsicLimits;
  yIntrinsicLimits = r.YIntrinsicLimits;
  xWorldLimits = r.XWorldLimits;
  yWorldLimits = r.YWorldLimits;
  xIntrinsic = xIntrinsicLimits(1) + (xWorld  - xWorldLimits(1)) ...
  / r.PixelExtentInWorldX;
  yIntrinsic = yIntrinsicLimits(1) + (yWorld  - yWorldLimits(1)) ...
  / r.PixelExtentInWorldY;
endfunction

%!error id=Octave:invalid-fun-call worldToIntrinsic (imref2d)
%!error id=Octave:invalid-fun-call worldToIntrinsic (imref2d, 1, 2, 3)
%!error id=Octave:expected-real worldToIntrinsic (imref2d, 1j, 2)
%!error id=Octave:expected-real worldToIntrinsic (imref2d, 1, 2j)
%!error id=Octave:invalid-input-arg worldToIntrinsic (imref2d, [1, 2], 3)
%!error id=Octave:invalid-input-arg worldToIntrinsic (imref2d, [1], [2, 3])

%!test
%! r = imref2d ([512, 512], 0.3125, 0.3125);
%! xW = [38.44, 39.44, 38.44, -0.2];
%! yW = [68.75, 68.75, 75.75, -1];
%! [xI, yI] = worldToIntrinsic (r, xW, yW);
%! assert (xI, [123.008, 126.208, 123.008, -0.64], 1e-6)
%! assert (yI, [220, 220, 242.4, -3.2], 1e-6)