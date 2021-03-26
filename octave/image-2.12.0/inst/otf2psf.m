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
## @deftypefn  {Function File} {} otf2psf (@var{otf})
## @deftypefnx {Function File} {} otf2psf (@var{otf}, @var{outsize})
## Compute PSF from OTF.
##
## Returns the Point Spread Function (OTF) of the Optical Transfer
## Function @var{otf}.
##
## The optional argument @var{outsize} defines the size of the returned
## @var{psf}.
##
## @seealso{circshift, ifft2, ifftn, psf2otf}
## @end deftypefn

function psf = otf2psf (otf, outsize)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (! isnumeric (otf))
    error ("otf2psf: OTF must be numeric")
  endif
  insize = size (otf);

  psf = ifftn (otf);
  psf = circshift (psf, floor (insize / 2));

  if (nargin > 1)
    if (! isnumeric (outsize) || ! isvector (outsize))
      error ("otf2psf: OUTSIZE must be a numeric vector");
    endif
    insize = size (otf);

    n = max (numel (outsize), numel (insize));
    outsize = postpad (outsize(:), n, 1);
    insize  = postpad (insize(:) , n, 1);

    pad = (insize - outsize) / 2;
    if (any (pad < 0))
      error ("otf2psf: OUTSIZE must be smaller than or equal than OTF size");
    endif
    prepad  = floor (pad);
    postpad = ceil  (pad);

    idx = arrayfun (@colon, prepad + 1, insize - postpad,
                    "UniformOutput", false);
    psf = psf(idx{:});
  endif

endfunction

## We are assuming psf2otf is working correctly

%!function otf = rand_otf (varargin)
%!  otf = complex (rand (varargin{:}), rand (varargin{:}));
%!endfunction

## Basic usage, 1, 2, and 3 dimensional
%!test
%! otf = rand_otf (6, 1);
%! assert (otf2psf (otf), circshift (ifft (otf), 3));
%!test
%! otf = rand_otf (6, 6);
%! assert (otf2psf (otf), circshift (ifft2 (otf), [3 3]));
%!test
%! otf = rand_otf (6, 6, 6);
%! assert (otf2psf (otf), circshift (ifftn (otf), [3 3 3]));

## Test when length of some sides are odd
%!test
%! otf = rand_otf (7, 1);
%! assert (otf2psf (otf), circshift (ifft (otf), 3));
%!test
%! otf = rand_otf (7, 7);
%! assert (otf2psf (otf), circshift (ifft2 (otf), [3 3]));
%!test
%! otf = rand_otf (6, 7, 8);
%! assert (otf2psf (otf), circshift (ifftn (otf), [3 3 4]));

## Test the outsize/unpadding option
%!test
%! otf  = rand_otf (7, 1);
%! ppsf = circshift (ifft (otf), 3);
%! assert (otf2psf (otf, 6), ppsf(1:6));
%! assert (otf2psf (otf, [6 1]), ppsf(1:6));
%!test
%! otf = rand_otf (7, 7);
%! ppsf = circshift (ifft2 (otf), [3 3]);
%! assert (otf2psf (otf, [6 1]), ppsf(1:6,4));
%!test
%! otf = rand_otf (6, 7);
%! ppsf = circshift (ifft2 (otf), [3 3]);
%! assert (otf2psf (otf, [6 6]), ppsf(:,1:6));

%!error <OTF must be numeric> otf2psf ("not a otf")
%!error <OUTSIZE must be smaller than> otf2psf (rand_otf (16), 18)
%!error <OUTSIZE must be smaller than> otf2psf (rand_otf (16), [14 18])
%!error <OUTSIZE must be smaller than> otf2psf (rand_otf (16), [18 18])
%!error <OUTSIZE must be smaller than> otf2psf (rand_otf (16, 1), 18)

## Some less random tests
%!test
%! psf = fspecial ("gaussian", 16);
%! otf = psf2otf (psf);
%! assert (otf2psf (otf), psf, eps);
%!test
%! psf = rand (16);
%! otf = psf2otf (psf);
%! assert (otf2psf (otf), psf, 4*eps);
%!test
%! psf = rand (8);
%! otf = psf2otf (psf, [16 16]);
%! assert (otf2psf (otf, [8 8]), psf, 2*eps);


