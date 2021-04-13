## Copyright (C) 2008 Søren Hauberg <soren@hauberg.org>
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
## @deftypefn {Function File} {@var{E} =} entropy (@var{im})
## @deftypefnx{Function File} {@var{E} =} entropy (@var{im}, @var{nbins})
## Computes the entropy of an image.
##
## The entropy of the elements of the image @var{im} is computed as
##
## @example
## @var{E} = -sum (@var{P} .* log2 (@var{P})
## @end example
##
## where @var{P} is the distribution of the elements of @var{im}. The distribution
## is approximated using a histogram with @var{nbins} cells. If @var{im} is
## @code{logical} then two cells are used by default. For other classes 256 cells
## are used by default.
##
## When the entropy is computed, zero-valued cells of the histogram are ignored.
##
## @seealso{entropyfilt}
## @end deftypefn

function retval = entropy (I, nbins = 0)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  endif

  if ( (! isnumeric (I) && ! islogical (I)) || issparse (I) || (! isreal (I)) )
    error ("entropy: I must be real, non-sparse and numeric");
  endif

  if (! isscalar (nbins))
    error ("entropy: NBINS must be a scalar");
  endif

  ## Get number of histogram bins
  if (nbins <= 0)
    if (islogical (I))
      nbins = 2;
    else
      nbins = 256;
    endif
  endif

  ## transform all non-logical images to uint8 (as Matlab):
  if (! islogical (I))
      I = im2uint8 (I);
  end

  ## Compute histogram, using imhist (as Matlab claims to do)
  P = imhist (I(:), nbins);

  ## ignore zero-entries of the histogram, and normalize it to a sum of 1
  P(P==0) = [];
  P = P ./ sum (P(:));

  ## Compute entropy
  retval = -sum (P .* log2 (P));
endfunction

%!assert (entropy ([0 1]), 1)
%!assert (entropy (uint8 ([0 1])), 1)
%!assert (entropy ([0 0]), 0)
%!assert (entropy ([0]), 0)
%!assert (entropy ([1]), 0)
%!assert (entropy ([0 .5; 2 0]), 1.5)

## rgb images are treated like nd grayscale images
%!assert (entropy (repmat ([0 .5; 2 0], 1, 1, 3)),
%!        entropy ([0 .5; 2 0]))

## test some 9x9 float input images
%!test
%! A = zeros (3,3);
%! B = ones (3,3);
%! C = [1 1 1; 2 2 2; 3 3 3];
%! D = C';
%! E = ones (3,3);
%! E(2,2)=2;
%! F = 3 .* ones (3,3);
%! F(2,2)=1;
%! G = [-1 2 7; -5 2 8; -7 pi 9];
%! H = [5 2 8; 1 -3 1; 5 1 0];
%! pG = [1 2] ./ 3;
%! G_out = -sum (pG.*log2 (pG));
%! pH = [2 7] ./ 9;
%! H_out = -sum (pH.*log2 (pH));
%! assert (entropy (A), 0, eps);
%! assert (entropy (B), 0, eps);
%! assert (entropy (C), 0, eps);
%! assert (entropy (D), 0, eps);
%! assert (entropy (E), 0, eps);
%! assert (entropy (F), 0, eps);
%! assert (entropy (G), G_out, eps);
%! assert (entropy (H), H_out, eps);

## test some 9x9 uint8 input images
%!test
%! A = uint8 (zeros (3,3));
%! B = uint8 (ones (3,3));
%! C = uint8 ([1 1 1; 2 2 2; 3 3 3]);
%! D = C';
%! E = uint8 (ones (3,3));
%! E(2,2)=2;
%! F = 3 .* uint8 (ones (3,3));
%! F(2,2)=1;
%! G = uint8 ([0 2 7; 0 2 8; 0 3 9]);
%! H = uint8 ([5 2 8; 1 0 1; 5 1 0]);
%! pC = [1 1 1] ./ 3;
%! C_out = -sum (pC.*log2 (pC));
%! D_out = C_out;
%! pE = [8 1] ./ 9;
%! E_out = -sum (pE.*log2 (pE));
%! F_out = E_out;
%! pG = [3 2 1 1 1 1] ./ 9;
%! G_out = -sum (pG.*log2 (pG));
%! pH = [2 3 1 2 1] ./ 9;
%! H_out = -sum (pH.*log2 (pH));
%! assert (entropy (A), 0);
%! assert (entropy (B), 0);
%! assert (entropy (C), C_out, eps);
%! assert (entropy (D), D_out, eps);
%! assert (entropy (E), E_out, eps);
%! assert (entropy (F), F_out, eps);
%! assert (entropy (G), G_out, eps);
%! assert (entropy (H), H_out, eps);

## test some 9x9 logical input images
%!test
%! L1 = false (3,3);
%! L1(2,2)=true;
%! L2 = true (3,3);
%! L2(2,2)=false;
%! L3 = logical ([0 1 1; 0 1 1; 0 0 1]);
%! p12 = [1 8] ./ 9;
%! out12 = -sum (p12.*log2 (p12));
%! p3 = [5 4] ./9;
%! out3 = -sum (p3.*log2 (p3));
%! assert (entropy (L1), out12, eps);
%! assert (entropy (L2), out12, eps);
%! assert (entropy (L3), out3, eps);
