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
## @deftypefn  {Function File} {} psf2otf (@var{psf})
## @deftypefnx {Function File} {} psf2otf (@var{psf}, @var{outsize})
## Compute OTF from PSF.
##
## Returns the Optical Transfer Function (OTF) of the Point Spread
## Function @var{psf}.
##
## The optional argument @var{outsize} defines the size of the computed
## @var{otf}.  The input @var{psf} is post-padded with zeros previous to
## the OTF computation.
##
## @seealso{circshift, fft2, fftn, otf2psf}
## @end deftypefn

function otf = psf2otf (psf, outsize)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (! isnumeric (psf) || ! isreal (psf))
    error ("psf2otf: PSF must be numeric and real")
  endif

  insize = size (psf);
  if (nargin > 1)
    if (! isnumeric (outsize) || ! isvector (outsize))
      error ("psf2otf: OUTSIZE must be a numeric vector");
    endif

    n = max (numel (outsize), numel (insize));
    outsize = postpad (outsize(:), n, 1);
    insize  = postpad (insize(:),  n, 1);
    if (any (outsize < insize))
      error ("psf2otf: OUTSIZE must be larger than or equal than PSF size");
    endif
    psf = padarray (psf, outsize - insize, "post", "zeros");
  endif

  psf = circshift (psf, - floor (insize / 2));
  otf = fftn (psf);

endfunction

## Basic usage, 1, 2, and 3 dimensional
%!test
%! psf = rand (6, 1);
%! assert (psf2otf (psf), fft (circshift (psf, [-3])));
%!test
%! psf = rand (6, 6);
%! assert (psf2otf (psf), fft2 (circshift (psf, [-3 -3])));
%!test
%! psf = rand (6, 6, 6);
%! assert (psf2otf (psf), fftn (circshift (psf, [-3 -3 -3])));

## Test when length of some sides are odd
%!test
%! psf = rand (7, 1);
%! assert (psf2otf (psf), fft (circshift (psf, [-3])));
%!test
%! psf = rand (7, 7);
%! assert (psf2otf (psf), fft2 (circshift (psf, [-3 -3])));
%!test
%! psf = rand (6, 7, 8);
%! assert (psf2otf (psf), fftn (circshift (psf, [-3 -3 -4])));

## Test the outsize/padding option
%!test
%! psf = rand (6, 1);
%! ppsf = [psf; 0];
%! assert (psf2otf (psf, 7), fft (circshift (ppsf, [-3])));
%!test
%! psf = rand (6, 1);
%! ppsf = [[psf; 0] zeros(7, 6)];
%! assert (psf2otf (psf, [7 7]), fft2 (circshift (ppsf, [-3 0])));
%!test
%! psf = rand (6, 6);
%! ppsf = [psf zeros(6, 1)];
%! assert (psf2otf (psf, [6 7]), fft2 (circshift (ppsf, [-3 -3])));

%!error <PSF must be numeric and real> psf2otf (complex (rand (16), rand (16)))
%!error <OUTSIZE must be larger than> psf2otf (rand (16), 14)
%!error <OUTSIZE must be larger than> psf2otf (rand (16), [14 14])
%!error <OUTSIZE must be larger than> psf2otf (rand (16), [18])
%!error <OUTSIZE must be larger than> psf2otf (rand (16), [18 14])

