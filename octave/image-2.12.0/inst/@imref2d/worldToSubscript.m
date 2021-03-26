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
## @deftypefn {} {[@var{i}, @var{j}] =} worldToSubscript (@var{r}, @var{xWorld}, @var{yWorld})
## Convert world coordinates to row and column subscripts.
##
## Converts world coordinates to row and column subscripts of an image
## associated with the spatial referencing object @var{r}.  A point located at
## (@var{xWorld}(i), @var{yWorld}(i)) world coordinates maps to row and column
## subscripts @var{i}(i) and @var{j}(i) respectively.  Note the reversed order
## of the dimensions.  If the point falls outside the bounds of the image, both
## its row and column subscripts are NaN.
##
## @seealso{imref2d, imref3d, worldToIntrinsic}
## @end deftypefn

function [i, j] = worldToSubscript (r, xWorld, yWorld)
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

  [xIntrinsic, yIntrinsic] = worldToIntrinsic (r, xWorld, yWorld);
  xIntrinsicLimits = r.XIntrinsicLimits;
  yIntrinsicLimits = r.YIntrinsicLimits;
  inImage = contains (r, xWorld, yWorld);
  xIntrinsic(! inImage) = NaN;
  yIntrinsic(! inImage) = NaN;
  i = round (yIntrinsic);
  j = round (xIntrinsic);
endfunction

%!error id=Octave:invalid-fun-call worldToSubscript (imref2d)
%!error id=Octave:invalid-fun-call worldToSubscript (imref2d, 1, 2, 3)
%!error id=Octave:expected-real worldToSubscript (imref2d, 1j, 2)
%!error id=Octave:expected-real worldToSubscript (imref2d, 1, 2j)
%!error id=Octave:invalid-input-arg worldToSubscript (imref2d, [1, 2], 3)
%!error id=Octave:invalid-input-arg worldToSubscript (imref2d, [1], [2, 3])

%!test
%! r = imref2d ([512, 512], 0.3125, 0.3125);
%! xW = [38.44, 39.44, 38.44, -0.2];
%! yW = [68.75, 68.75, 75.75, -1];
%! [rS, cS] = worldToSubscript (r, xW, yW);
%! assert (rS, [220, 220, 242, NaN])
%! assert (cS, [123, 126, 123, NaN])