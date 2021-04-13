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
## @deftypefn {} {[@var{xWorld}, @var{yWorld}] =} intrinsicToWorld (@var{r}, @var{xIntrinsic}, @var{yIntrinsic})
## Convert from intrinsic to world coordinates.
##
## Converts intrinsic coordinates of the image associated with the spatial
## referencing object @var{r} to world coordinates @var{xWorld}
## and @var{yWorld}.  If a point (@var{xIntrinsic}(i), @var{yIntrinsic}(i))
## falls outside the intrinsic bounds of the image, the world coordinates are
## extrapolated, possibly resulting in negative values.
##
## @seealso{imref2d, imref3d, worldToIntrinsic}
## @end deftypefn

function [xWorld, yWorld] = intrinsicToWorld (r, xIntrinsic, yIntrinsic)
  if (nargin != 3)
    print_usage ();
  endif
  
  validateattributes (xIntrinsic, {"numeric"}, ...
  {"real"}, "imref2d", "xIntrinsic");
  validateattributes (yIntrinsic, {"numeric"}, ...
  {"real"}, "imref2d", "yIntrinsic");
  
  if (! all (size (xIntrinsic) == size (yIntrinsic)))
    error ("Octave:invalid-input-arg", ...
    "xIntrinsic and yIntrinsic must be of the same size");
  endif
  
  xIntrinsicLimits = r.XIntrinsicLimits;
  yIntrinsicLimits = r.YIntrinsicLimits;
  xWorldLimits = r.XWorldLimits;
  yWorldLimits = r.YWorldLimits;
  xWorld = xWorldLimits(1) + r.PixelExtentInWorldX * ...
  (xIntrinsic - xIntrinsicLimits(1));
  yWorld = yWorldLimits(1) + r.PixelExtentInWorldY * ...
  (yIntrinsic - yIntrinsicLimits(1));
endfunction

%!error id=Octave:invalid-fun-call intrinsicToWorld (imref2d)
%!error id=Octave:invalid-fun-call intrinsicToWorld (imref2d, 1, 2, 3)
%!error id=Octave:expected-real intrinsicToWorld (imref2d, 1j, 2)
%!error id=Octave:expected-real intrinsicToWorld (imref2d, 1, 2j)
%!error id=Octave:invalid-input-arg intrinsicToWorld (imref2d, [1, 2], 3)
%!error id=Octave:invalid-input-arg intrinsicToWorld (imref2d, [1], [2, 3])

%!test
%! r = imref2d ([512, 512], 0.3125, 0.3125);
%! xIntrinsic = [34, 442];
%! yIntrinsic = [172, 172];
%! [xWorld, yWorld] = intrinsicToWorld (r, xIntrinsic, yIntrinsic);
%! assert (xWorld, [10.625, 138.125])
%! assert (yWorld, [53.75, 53.75])

%!test
%! [xWorld, yWorld] = intrinsicToWorld (imref2d, -5.3, -2.8);
%! assert (xWorld, -5.3)
%! assert (yWorld, -2.8)

%!test
%! [xW, yW] = intrinsicToWorld (imref2d, [1, 2; 3, 4],  [2, 3; 5, 9]);
%! assert (xW, [1, 2; 3, 4])
%! assert (yW, [2, 3; 5, 9])