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
## @deftypefn {} {@var{tf} =} contains (@var{r}, @var{xWorld}, @var{yWorld}, @var{zWorld})
## Determine if image contains points in world coordinate system.
##
## Outputs a logical array @var{tf}, where the i-th nonzero value means the
## point (@var{xWorld}(i), @var{yWorld}(i), @var{zWorld}(i)) lies within the
## bounds of an image associated with a spatial referencing object @var{r}.
##
## @seealso{imref2d, imref3d}
## @end deftypefn

function tf = contains (r, xWorld, yWorld, zWorld)
  if (nargin != 4)
    print_usage();
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
    "imref3d/contains: xWorld, yWorld and zWorld must be of the same size");
  endif
  
  xWorldLimits = r.XWorldLimits;
  yWorldLimits = r.YWorldLimits;
  zWorldLimits = r.ZWorldLimits;
  containsX = xWorld >= xWorldLimits(1) & xWorld <= xWorldLimits(2);
  containsY = yWorld >= yWorldLimits(1) & yWorld <= yWorldLimits(2);
  containsZ = zWorld >= zWorldLimits(1) & zWorld <= zWorldLimits(2);
  tf = containsX & containsY & containsZ;
endfunction

%!error id=Octave:invalid-fun-call contains (imref3d)
%!error id=Octave:invalid-fun-call contains (imref3d, 1)
%!error id=Octave:invalid-fun-call contains (imref3d, 1, 2)
%!error id=Octave:invalid-fun-call contains (imref3d, 1, 2, 3, 4)
%!error id=Octave:invalid-input-arg contains (imref3d, [1, 2], 3, 4)
%!error id=Octave:invalid-input-arg contains (imref3d, 1, [2, 3], 4)
%!error id=Octave:invalid-input-arg contains (imref3d, 1, 2, [3, 4])
%!error id=Octave:expected-real contains (imref3d, 1j, 2, 3)
%!error id=Octave:expected-real contains (imref3d, 1, 2j, 3)
%!error id=Octave:expected-real contains (imref3d, 1, 2, 3j)

%!test
%! r = imref3d ([128, 128, 27]);
%! assert (contains (r, [5, 6, 6, 8], [5, 10, 10, 257], [1, 27.5, 28, 1]), logical ([1, 1, 0, 0]))