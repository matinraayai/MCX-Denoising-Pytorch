## Copyright (C) 2006 Søren Hauberg <soren@hauberg.org>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} @var{warped} = imremap(@var{im}, @var{XI}, @var{YI})
## @deftypefnx{Function File} @var{warped} = imremap(@var{im}, @var{XI}, @var{YI}, @var{interp}, @var{extrapval})
## Applies any geometric transformation to the image @var{im}.
##
## The arguments @var{XI} and @var{YI} are lookup tables that define the resulting
## image
## @example
## @var{warped}(y,x) = @var{im}(@var{YI}(y,x), @var{XI}(y,x))
## @end example
## where @var{im} is assumed to be a continuous function, which is achieved
## by interpolation. Note that the image @var{im} is expressed in a (X, Y)-coordinate
## system and not a (row, column) system.
##
## The optional argument @var{method} defines the interpolation method to be
## used.  All methods supported by @code{interp2} can be used.  By default, the
## @code{linear} method is used.
##
## For @sc{matlab} compatibility, the methods @code{bicubic} (same as
## @code{cubic}), @code{bilinear} and @code{triangle} (both the same as
## @code{linear}) are also supported.
##
## All values of the result that fall outside the original image will
## be set to @var{extrapval}.  The default value of @var{extrapval} is 0.
##
## @seealso{imperspectivewarp, imrotate, imresize, imshear, interp2}
## @end deftypefn

function [warped] = imremap(im, XI, YI, interp = "linear", extrapval = 0)

  if (nargin < 3 || nargin > 5)
    print_usage ();
  elseif (! isimage (im) || ndims (im) > 3)
    error ("imremap: IM must be a grayscale or RGB image.")
  elseif (! size_equal (XI, YI) || ! ismatrix (XI) || ! isnumeric (XI))
    error ("imremap: XI and YI must be matrices of the same size");
  elseif (! ischar (interp))
    error ("imremap: INTERP must be a string with interpolation method")
  elseif (! isscalar (extrapval))
    error ("imremap: EXTRAPVAL must be a scalar");
  endif
  interp = interp_method (interp);

  ## Interpolate
  sz = size (im);
  n_planes = prod (sz(3:end));
  sz(1:2) = size (XI);
  warped = zeros(sz);
  for i = 1:n_planes
    warped(:,:,i) = interp2 (double(im(:,:,i)), XI, YI, interp, extrapval);
  endfor

  ## we return image on same class as input
  warped = cast (warped, class (im));

endfunction


%!demo
%! ## Generate a synthetic image and show it
%! I = tril(ones(100)) + abs(rand(100)); I(I>1) = 1;
%! I(20:30, 20:30) = !I(20:30, 20:30);
%! I(70:80, 70:80) = !I(70:80, 70:80);
%! figure, imshow(I);
%! ## Resize the image to the double size and show it
%! [XI, YI] = meshgrid(linspace(1, 100, 200));
%! warped = imremap(I, XI, YI);
%! figure, imshow(warped);

%!demo
%! ## Generate a synthetic image and show it
%! I = tril(ones(100)) + abs(rand(100)); I(I>1) = 1;
%! I(20:30, 20:30) = !I(20:30, 20:30);
%! I(70:80, 70:80) = !I(70:80, 70:80);
%! figure, imshow(I);
%! ## Rotate the image around (0, 0) by -0.4 radians and show it
%! [XI, YI] = meshgrid(1:100);
%! R = [cos(-0.4) sin(-0.4); -sin(-0.4) cos(-0.4)];
%! RXY = [XI(:), YI(:)] * R;
%! XI = reshape(RXY(:,1), [100, 100]); YI = reshape(RXY(:,2), [100, 100]);
%! warped = imremap(I, XI, YI);
%! figure, imshow(warped);
