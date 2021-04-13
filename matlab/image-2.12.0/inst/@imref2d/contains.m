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
## @deftypefn {} {@var{tf} =} contains (@var{r}, @var{xWorld}, @var{yWorld})
## Determine if image contains points in world coordinate system.
##
## Outputs a logical array @var{tf}, where the i-th nonzero value means the
## point (@var{xWorld}(i), @var{yWorld}(i)) lies within the bounds of
## an image associated with a spatial referencing object @var{r}.
##
## @seealso{imref2d, imref3d}
## @end deftypefn

function tf = contains (r, xWorld, yWorld)
  if (nargin != 3)
    print_usage();
  endif

  validateattributes (xWorld, {"numeric"}, ...
  {"real"}, "imref2d", "xWorld");
  validateattributes (yWorld, {"numeric"}, ...
  {"real"}, "imref2d", "yWorld");
  
  if (! all (size (xWorld) == size (yWorld)))
    error ("Octave:invalid-input-arg", ...
    "xWorld and yWorld must be of the same size");
  endif

  xWorldLimits = r.XWorldLimits;
  yWorldLimits = r.YWorldLimits;
  containsX = xWorld >= xWorldLimits(1) & xWorld <= xWorldLimits(2);
  containsY = yWorld >= yWorldLimits(1) & yWorld <= yWorldLimits(2);
  tf = containsX & containsY;
endfunction

%!error id=Octave:invalid-fun-call contains (imref2d)
%!error id=Octave:invalid-fun-call contains (imref2d, 1)
%!error id=Octave:invalid-fun-call contains (imref2d, 1, 2, 3)
%!error id=Octave:invalid-input-arg contains (imref2d, 1, [2, 3])
%!error id=Octave:invalid-input-arg contains (imref2d, [1, 2], 3)
%!error id=Octave:expected-real contains (imref2d, 0, j)
%!error id=Octave:expected-real contains (imref2d, j, 0)

%!assert (contains (imref2d, [], []), logical( zeros (0, 0)))
%!assert (contains (imref2d, [1, 2; 3, 4], [5, -6; 7, 8]), logical (zeros (2, 2)))

%!test
%! r = imref2d ([256, 256]);
%! assert (contains(r, [5, 8, 8], [5, 10, 257]), logical([1, 1, 0]))