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
## @deftypefn {} {@var{TF} =} sizesMatch (@var{r}, @var{A})
## Determine if object and image are size-compatible.
##
## Outputs logical 1 (true) if the first two dimensions of an n-dimensional
## image @var{A} match the image size of the spatial referencing object @var{r},
## otherwise outputs zero (false).
##
## @seealso{imref2d, imref3d}
## @end deftypefn

function TF = sizesMatch (r, A)
  if (nargin != 2)
    print_usage();
  endif

  sizeA = size(A);
  TF = all(sizeA(1:2) == r.ImageSize);
endfunction

%!error id=Octave:invalid-fun-call sizesMatch (imref2d)

## example from MATLAB documentation
%!test
%! I = zeros (256, 256);
%! r = imref2d ([256, 256]);
%! assert (sizesMatch (r, I), true)
%! I2 = zeros (246, 300);
%! assert (sizesMatch (r, I2), false)

## accepts empty image
%!test
%! r = imref2d ([256, 256]);
%! assert (sizesMatch (r, []), false)

## accepts 1-D image
%!test
%! r = imref2d ([256, 256]);
%! assert (sizesMatch (r, 42), false)

## accepts N-D image
%!test
%! r = imref2d ([256, 256]);
%! assert (sizesMatch (r, zeros (256, 256, 3, 2)), true)

%!test
%! I = zeros (384, 512, 3);
%! r = imref2d (size (I));
%! assert (sizesMatch (r, I), true)