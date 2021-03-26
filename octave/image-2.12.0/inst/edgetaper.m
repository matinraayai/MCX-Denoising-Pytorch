## Copyright (C) 2015 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} edgetaper (@var{img}, @var{psf})
## Blur border (edges) of image to prevent ringing artifacts.
##
## @emph{Warning}: this function is not @sc{Matlab} compatible and
## is likely to change in the future.
##
## @end deftypefn

function imgt = edgetaper (img, psf)
  if (nargin != 2)
    print_usage ();
  elseif (! isnumeric (img))
    error ("edgetaper: IMG must be numeric")
  elseif (! isnumeric (psf))
    error ("edgetaper: PSF must be numeric")
  endif

  img_size = size (img);
  psf_size = size (psf);
  n = max (numel (img_size), numel (psf_size));
  img_size = postpad (img_size(:), n, 1);
  psf_size = postpad (psf_size(:), n, 1);

  ## beware of psf_size = [16 16 1 8] paired with img_size [512 512 1 50]
  ## which are valid, singleton dimensions do not count for this check
  if (any ((psf_size > (img_size / 2)) & (psf_size > 1)))
    error ("edgetaper: PSF must be smaller than half of IMG dimensions")
  endif
  psf = psf ./ sum (psf(:));  # we use it for blurring so the sum must be 1

  ## FIXME  this function is not Matlab compatible.  I have no clue what
  ##        Matlab is doing but is definitely not what they claim on the
  ##        documentation.  There are no references for the function and
  ##        the documentation is sparse and incorrect.
  ##
  ##        I have found papers that compare their method for reducing
  ##        boundary artifacts against Matlab's edgetaper but even they
  ##        do not comment it.
  ##
  ##        It an implementation of edgetaper that is close to Matlab's
  ##        documentation for the function.  If anyone has patience, please
  ##        fix this.
  ##
  ##        Some questions about it:
  ##
  ##          1. the autocorrelation of the PSF is twice the size of the PSF
  ##          but it will still be way smaller than the image.  How can it be
  ##          used to make a weighted mean between the blurred and the original
  ##          image? We pretty much split it and only use it on the borders
  ##          but looks like we end up with an image too blurred.
  ##
  ##          2. how do they blur the image? I will guess they pad the
  ##          image but how?
  ##
  ##  Note: always test this function with input as double precision
  ##        since it returns the same class as input and may hide our
  ##        differences.

  blurred = fftconvn (img, psf, "same");
  xpsf = normalized_autocorrelation (psf);

  ## The following will expand a ND matrix into a larger size
  ## repeating the center elements, e.g.,
  ##
  ##              1 2 2 2 3
  ##  1 2 3       4 5 5 5 6
  ##  4 5 6   =>  4 5 5 5 6
  ##  7 8 9       4 5 5 5 6
  ##              7 8 8 8 9

  ## note the xpsf will always have odd sizes (2*psf_size +1)
  xdims = ndims (xpsf);
  xpsf_size = size (xpsf)(:);
  idim = arrayfun (@(c, n, f) [1:c repmat(c, [1 n]) c:f], ceil (xpsf_size /2),
                   img_size(1:xdims) - xpsf_size -1, xpsf_size,
                  "UniformOutput", false);
  subs = cell (xdims, 1);
  [subs{:}] = ndgrid (idim{:});
  inds = sub2ind (xpsf_size, subs{:});
  weights = xpsf(inds);

  imgt = (img .* weights) + (blurred .* (1 - weights));

  imgt = cast (imgt, class (img));
endfunction

function acn = normalized_autocorrelation (psf)
  idx = arrayfun (@colon, size (psf), repmat (-1, [1 ndims(psf)]),
                  repmat (1, [1 ndims(psf)]), "UniformOutput", false);
  ac = convn (psf, conj (psf(idx{:})));
  acn = ac / max (ac(:));
endfunction

%!assert (class (edgetaper (rand (100), rand (16))), "double")
%!assert (class (edgetaper (randi (255, 100, "uint8"), rand (16))), "uint8")

