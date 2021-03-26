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
## @deftypefn {Function File} axes2pix {@var{pixelCoord} =} axes2pix (@var{n}, @var{extent}, @var{axesCoord})
## Convert axes coordinates to pixel coordinates.
##
## Converts coordinates @var{axesCoord} on a single axis in world units to
## intrinsic pixel coordinates of a 2-D image with @var{n} rows or columns,
## given a two-element real vector @var{extent} representing coordinates
## of centers of the first and last pixel in world units.  Intrinsic pixel
## coordinates are always x = 1.0, y = 1.0 in the center of a top left pixel
## and x = number of columns, y = number of rows in the center of a bottom
## right pixel and are continuous.  The actual coordinate range spanned by
## the image is larger than @var{exntent} by half a pixel on each side.
## For example if @var{extent} is [1, 2] and @var{n} is 2, the coordinate range
## spanned by the image on the axis is [0.5, 2.5].
##
## Elements of the real vector @var{axesCoord} are trated as individual points
## on the axis. @var{extent} can also be given in reverse order.
##
## MATLAB compatibility note: in order to produce results similar to MATLAB,
## very few argument validity checks are done. Octave will accept any scalar
## @var{n} (only positive integers make any sense), more-than-two-element vector
## @var{extent} (in which case only the first and last element are used),
## coordinates outside the the coordinate range spanned by the image
## (possibly resulting in negative pixel coordinates) or any matrix
## @var{axesCoord}. It's up to the caller to provide sensible inputs.
##
## @example
## @group
## ## 800x600 pixel image with its top left pixel centered at (100, 0)
## xData = [100, 140];
## yData = [0, 30];
## axes2pix(800, xData, [100, 120, 140])
## @result{} 1.0 400.5 800.0
## axes2pix(600, yData, [0, 15, 30])
## @result{} 1.0 300.50 600.0
## @end group
## @end example
##
## @example
## @group
## ## x-axis reversed
## xData = [140, 100];
## axes2pix(800, xData, [100, 120, 140])
## @result{} 800.0 400.5 1.0
## @end group
## @end example
##
## @end deftypefn

function pixelCoord = axes2pix (n, extent, axesCoord)
  if (nargin != 3)
    print_usage ();
  endif

  if (! isscalar (n))
    error ("Octave:invalid-input-arg", "axes2pix: N must be a scalar");
  endif

  if (! isvector (extent))
    error ("Octave:invalid-input-arg", "axes2pix: EXTENT must be a vector");
  endif

  if ((n == 1) || extent(1) == extent(end))
    pixelCoord = axesCoord - extent(1) + 1;
  else
    pixelWidth = (extent(end) - extent(1)) ./ (n - 1);
    pixelCoord = (axesCoord - extent(1)) ./ pixelWidth + 1;
  endif
endfunction

## test argument checking
%!error id=Octave:invalid-fun-call axes2pix ()
%!error id=Octave:invalid-fun-call axes2pix (42)
%!error id=Octave:invalid-fun-call axes2pix (42, [1, 2])
%!error id=Octave:invalid-input-arg axes2pix ([42, 43], [1, 2], [1, 2, 3])
%!error id=Octave:invalid-input-arg axes2pix (42, [1, 2; 3, 4], [1, 2, 3])

## empty input
%!assert (axes2pix (42, [1 42], []), [])

## some numbers from MATLAB axes2pix documentation
%!assert (axes2pix (240, [1, 240], 30), 30)
%!assert (axes2pix (291, [1, 291], 30), 30)
%!assert (axes2pix (240, [400.5, 520], 450), 100)
%!assert (axes2pix (291, [-19, 271], 90), 110)

## world width is zero
%!assert (axes2pix (1, [1 1], [0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
%!assert (axes2pix (5, [1 1], [0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])
%!assert (axes2pix (0, [1 1], [0, 1, 2, 3, 4]), [0, 1, 2, 3, 4])

## world axes are in reverse order
%!assert (axes2pix (5, [5 1], [1, 2, 3, 4, 5]), [5, 4, 3, 2, 1])
%!assert (axes2pix (5, [3 -1], [1, 2, 3, 4, 5]), [3, 2, 1, 0, -1])
%!assert (axes2pix (25, [5 1], [1, 2, 3, 4, 5]), [25, 19, 13, 7, 1])

## single row/column
%!assert (axes2pix (1, [1 5], [1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])
%!assert (axes2pix (1, [5 1], [-1, 0, 1, 2.5]), [-5, -4, -3, -1.5])
%!assert (axes2pix (1, [-10 -15], [-1, 0, 1.5]), [10, 11, 12.5])

## extent as a column vector
%!assert (axes2pix (5, [5; 1], [1, 2, 3, 4, 5]), [5, 4, 3, 2, 1])

## axesCoord as a column vector
%!assert (axes2pix (5, [5; 1], [1; 2; 3; 4; 5]), [5; 4; 3; 2; 1])
