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
## @deftypefn {} {[@var{i}, @var{j}, @var{k}] =} worldToSubscript (@var{r}, @var{xWorld}, @var{yWorld}, @var{zWorld})
## Convert world coordinates to row, column and plane subscripts.
##
## Converts world coordinates to row, column and plane subscripts of an image
## associated with the spatial referencing object @var{r}.  A point located at
## (@var{xWorld}(i), @var{yWorld}(i), @var{zWorld}(i)) world coordinates maps
## to row, column and plane subscripts @var{i}(i), @var{j}(i), @var{k}(i)
## respectively.  Note the reversed order of the first two dimensions.  If the
## point falls outside the bounds of the image, all of its subscripts are NaN.
##
## @seealso{imref2d, imref3d, worldToIntrinsic}
## @end deftypefn

function [i, j, k] = worldToSubscript (r, xWorld, yWorld, zWorld)
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

  [xIntrinsic, yIntrinsic, zIntrinsic] ...
  = worldToIntrinsic (r, xWorld, yWorld, zWorld);
  xIntrinsicLimits = r.XIntrinsicLimits;
  yIntrinsicLimits = r.YIntrinsicLimits;
  zIntrinsicLimits = r.ZIntrinsicLimits;
  inImage = contains (r, xWorld, yWorld, zWorld);
  xIntrinsic(! inImage) = NaN;
  yIntrinsic(! inImage) = NaN;
  zIntrinsic(! inImage) = NaN;
  i = round (yIntrinsic);
  j = round (xIntrinsic);
  k = round (zIntrinsic);
endfunction

%!error id=Octave:invalid-fun-call worldToSubscript (imref3d)
%!error id=Octave:invalid-fun-call worldToSubscript (imref3d, 1)
%!error id=Octave:invalid-fun-call worldToSubscript (imref3d, 1, 2)
%!error id=Octave:invalid-fun-call worldToSubscript (imref3d, 1, 2, 3, 4)
%!error id=Octave:expected-real worldToSubscript (imref3d, 1j, 2, 3)
%!error id=Octave:expected-real worldToSubscript (imref3d, 1, 2j, 3)
%!error id=Octave:expected-real worldToSubscript (imref3d, 1, 2, 3j)
%!error id=Octave:invalid-input-arg worldToSubscript (imref3d, [1, 2], 3, 4)
%!error id=Octave:invalid-input-arg worldToSubscript (imref3d, 1, [2, 3], 4)
%!error id=Octave:invalid-input-arg worldToSubscript (imref3d, 1, 2, [3, 4])

%!test
%! r = imref3d ([128, 128, 27], 2, 2, 4);
%! xW = [108, 108, 113.2, 2];
%! yW = [92, 92, 92, -1];
%! zW = [52, 55, 52, 0.33];
%! [rS, cS, pS] = worldToSubscript (r, xW, yW, zW);
%! assert (rS, [46, 46, 46, NaN])
%! assert (cS, [54, 54, 57, NaN])
%! assert (pS, [13, 14, 13, NaN])