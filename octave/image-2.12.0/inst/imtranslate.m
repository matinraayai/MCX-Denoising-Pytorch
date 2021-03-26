## Copyright (C) 2002 Jeff Orchard <jjo@cs.sfu.ca>
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
## @deftypefn {Function File} {@var{Y}} = imtranslate (@var{M}, @var{x}, @var{y})
## @deftypefnx {Function File} {@var{Y}} = imtranslate (@var{M}, @var{x}, @var{y}, @var{bbox})
## Translate a 2D image by (x,y) using Fourier interpolation.
##
## @var{M} is a matrix, and is translated to the right by @var{X} pixels
## and translated up by @var{Y} pixels.
##
## @var{bbox} can be either 'crop' or 'wrap' (default).
##
## @end deftypefn

function Y = imtranslate (X, nc, nr, bbox = "wrap")

  if (strcmp (bbox, "crop"))
    pre = post = [0 0];
    if (nc > 0)
      post(2) = ceil (nc);
    else
      pre(2) = ceil (nc);
    endif
    if (nr > 0)
      pre(1) = ceil (nr);
    else
      post(1) = ceil (nr);
    endif
    pre = abs (pre);
    post = abs (post);
    X = padarray (X, abs (pre), "pre");
    X = padarray (X, abs (post), "post");
  endif

  [dimy, dimx] = size(X);
  x = fft2(X);
  px = exp(-2*pi*i*nc*(0:dimx-1)/dimx);
  py = exp(-2*pi*i*nr*(0:dimy-1)/dimy)';   % actually to correspond to index notation 'b' should be
                                          % replaced with '-b'
                                          % but I do not want to brake previous version compatibility
                                          % note: it also must be done in the cropping iand padding code
  P = py * px;
  y = x .* P;
  Y = real(ifft2(y));   % fft return complex number
                        % for integer shifts imaginary part  is 0 
                        % so real takes care of transfer from complex number to real

  if (strcmp (bbox, "crop"))
    Y = Y(pre(1)+1:dimy-post(1) , pre(2)+1:dimx-post(2));
  endif
endfunction

%!test
%! obs = imtranslate (ones (5, 5), 2, 1, "crop");
%! exp = zeros (5, 5);
%! exp(1:4, 3:5) = 1;
%! assert (obs, exp, eps * 10)
%!
%! obs = imtranslate (ones (5, 5), -2, -1, "crop");
%! exp = zeros (5, 5);
%! exp(2:5, 1:3) = 1;
%! assert (obs, exp, eps * 10)
