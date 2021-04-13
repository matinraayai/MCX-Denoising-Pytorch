## Copyright (C) 2013 Brandon Miles <brandon.miles7@gmail.com>
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn  {Function File} {[@var{gradX}, @var{gradY}] =} imgradientxy (@var{img})
## @deftypefnx {Function File} {[@var{gradX}, @var{gradY}] =} imgradientxy (@var{img}, @var{method})
## Compute the x and y gradients of an image using various methods.
##
## The first input @var{img} is the gray scale image to compute the edges
## on.  The second input @var{method} controls the method used to calculate
## the gradients.
##
## The first output @var{gradX} returns the gradient in the x direction.
## The second output @var{gradY} returns the gradient in the y direction.
##
## The @var{method} input argument can be any of the following strings:
##
## @table @asis
## @item @qcode{"sobel"} (default)
## Calculates the gradient using the Sobel approximation to the
## derivatives.
##
## @item @qcode{"prewitt"}
## Calculates the gradient using the Prewitt approximation to the
## derivatives.  This method works just like Sobel except a different
## approximation of the gradient is used.
##
## @item @qcode{"central"} or @qcode{"centraldifference"}
## Calculates the gradient using the central difference approximation to the
## derivatives: @code{(x(i-1) - x(i+1))/2}.
##
## @item @qcode{"intermediate"} or @qcode{"intermediatedifference"}
## Calculates the gradient in using the intermediate difference approximation
## to the derivatives: @code{x(i) - x(i+1)}.
##
## @end table
##
## @seealso{edge, imgradient}
## @end deftypefn

function [gradX, gradY] = imgradientxy (img, method = "sobel")

  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (! isgray (img))
    error ("imgradientxy: IMG must be a gray-scale image");
  elseif (! ischar (method))
    error ("imgradientxy: METHOD must be a string");
  endif

  switch (tolower (method))
    case {"sobel", "prewitt"}
      ker = fspecial (method); # horizontal

    case {"central", "centraldifference"}
      ker = [0.5; 0; -0.5];

    case {"intermediate", "intermediatedifference"}
      ker = [1; -1];

    otherwise
      error("imgradientxy: unrecognized METHOD %s", method);
  endswitch

  gradX = conv2 (img, ker', "same");
  if (nargout > 1)
    gradY = conv2 (img, ker, "same");
  endif

endfunction

%!test
%! A = [0 1 0
%!      1 1 1
%!      0 1 0];
%!
%! [gxSobel,  gySobel]  = imgradientxy (A);
%! [gxSobel2, gySobel2] = imgradientxy (A, "Sobel");
%! assert (gxSobel,
%!         [ 3   0  -3
%!           4   0  -4
%!           3   0  -3]);
%! assert (gySobel,
%!         [ 3   4   3
%!           0   0   0
%!          -3  -4  -3]);
%!
%! ## test default method
%! assert(gxSobel, gxSobel2);
%! assert(gySobel, gySobel2);
%!
%! [gxPrewitt, gyPrewitt] = imgradientxy (A, "Prewitt");
%! assert (gxPrewitt,
%!         [ 2   0  -2
%!           3   0  -3
%!           2   0  -2]);
%! assert (gyPrewitt,
%!         [ 2   3   2
%!           0   0   0
%!          -2  -3  -2]);
%!
%! [gxCd, gyCd] = imgradientxy (A, "CentralDifference");
%! assert (gxCd,
%!         [ 0.5   0.0  -0.5
%!           0.5   0.0  -0.5
%!           0.5   0.0  -0.5]);
%! assert (gyCd,
%!         [ 0.5   0.5   0.5
%!           0     0     0
%!          -0.5  -0.5  -0.5]);
%!
%! [gxCd, gyCd] = imgradientxy (A, "Central");
%! assert (gxCd,
%!         [ 0.5   0.0  -0.5
%!           0.5   0.0  -0.5
%!           0.5   0.0  -0.5]);
%! assert (gyCd,
%!         [ 0.5   0.5   0.5
%!           0     0     0
%!          -0.5  -0.5  -0.5]);
%!
%! [gxId, gyId] = imgradientxy(A, "IntermediateDifference");
%! assert (gxId,
%!         [ 1  -1   0
%!           0   0  -1
%!           1  -1   0]);
%! assert (gyId,
%!         [ 1   0   1
%!          -1   0  -1
%!           0  -1   0]);
%!
%! [gxId, gyId] = imgradientxy(A, "Intermediate");
%! assert (gxId,
%!         [ 1  -1   0
%!           0   0  -1
%!           1  -1   0]);
%! assert (gyId,
%!         [ 1   0   1
%!          -1   0  -1
%!           0  -1   0]);

