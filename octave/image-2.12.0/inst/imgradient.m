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
## @deftypefn  {Function File} {[@var{gradMag}, @var{gradDir}] =} imgradient (@var{img})
## @deftypefnx {Function File} {[@var{gradMag}, @var{gradDir}] =} imgradient (@var{img}, @var{method})
## @deftypefnx {Function File} {[@var{gradMag}, @var{gradDir}] =} imgradient (@var{gx}, @var{gy})
## Compute the gradient magnitude and direction in degrees for an image.
##
## These are computed from the @var{gx} and @var{xy} gradients using
## @code{imgradientxy}.  The first input @var{img} is a gray scale image to
## compute the edges on.  The second input @var{method} controls the method
## used to calculate the gradients.  Alternatively the first input @var{gx}
## can be the x gradient and the second input @var{gy} can be the y gradient.
##
## The first output @var{gradMag} returns the magnitude of the gradient.
## The second output @var{gradDir} returns the direction in degrees.
##
## The @var{method} input argument must be a string specifying one of the
## methods supported by @code{imgradientxy}.
##
## @seealso{edge, imgradientxy}
## @end deftypefn

function [gradMag, gradDir] = imgradient (img, method = "sobel")

  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (ndims (img) != 2)
    error("imgradient: IMG must be a 2 dimensional matrix");
  endif

  if (ischar (method))
    [gradX, gradY] = imgradientxy (img, method);
  else
    ## we already got gX and gY, just confirm it's good data
    if (! size_equal (img, method))
      error("imgradient: GX and GY must be of equal size")
    endif
    gradX = img;
    gradY = method;
  endif

  gradMag = sqrt (gradX.^2 + gradY.^2);

  if (nargout > 1)
    ## Why imgradient invert vertical when computing the angle
    ## Use atan2(-gy,gx)*pi/180. See http://stackoverflow.com/questions/18549015
    gradDir = atan2d (-gradY, gradX);
  endif

endfunction

%!test
%! A = [0 1 0
%!      1 1 1
%!      0 1 0];
%!
%! [gMag, gDir] = imgradient (A);
%! assert (gMag,[sqrt(18) 4 sqrt(18); 4 0 4; sqrt(18),4,sqrt(18)]);
%! assert (gDir,[-45 -90 -135; -0 -0 -180; 45 90 135]);
%!
%! ## the following just test if passing gx and gy separately gets
%! ## us the same as the image and method though imgradient
%! [gxSobel, gySobel] = imgradientxy (A, "Sobel");
%! [gxPrewitt, gyPrewitt] = imgradientxy (A, "Prewitt");
%! [gxCd, gyCd] = imgradientxy (A, "CentralDifference");
%! [gxId, gyId] = imgradientxy (A, "IntermediateDifference");
%!
%! assert (imgradient (A),
%!         imgradient (gxSobel, gySobel));
%! assert (imgradient (A, "Sobel"),
%!         imgradient (gxSobel, gySobel));
%! assert (imgradient (A, "Prewitt"),
%!         imgradient(gxPrewitt, gyPrewitt));
%! assert (imgradient (A, "CentralDifference"),
%!         imgradient (gxCd, gyCd));
%! assert (imgradient (A, "IntermediateDifference"),
%!         imgradient (gxId, gyId));

