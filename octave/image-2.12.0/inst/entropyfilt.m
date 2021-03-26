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
## @deftypefn {Function File} {@var{E} =} entropyfilt (@var{im})
## @deftypefnx{Function File} {@var{E} =} entropyfilt (@var{im}, @var{domain})
## @deftypefnx{Function File} {@var{E} =} entropyfilt (@var{im}, @var{domain}, @var{padding}, @dots{})
## Computes the local entropy in a neighbourhood around each pixel in an image.
##
## The entropy of the elements of the neighbourhood is computed as
##
## @example
## @var{E} = -sum (@var{P} .* log2 (@var{P})
## @end example
##
## where @var{P} is the distribution of the elements of @var{im}. The distribution
## is approximated using a histogram with @var{nbins} cells. If @var{im} is
## @code{logical} then two cells are used. For other classes 256 cells
## are used.
##
## When the entropy is computed, zero-valued cells of the histogram are ignored.
##
## The neighbourhood is defined by the @var{domain} binary mask. Elements of the
## mask with a non-zero value are considered part of the neighbourhood. By default
## a 9 by 9 matrix containing only non-zero values is used.
##
## At the border of the image, extrapolation is used. By default symmetric
## extrapolation is used, but any method supported by the @code{padarray} function
## can be used. Since extrapolation is used, one can expect a lower entropy near
## the image border.
##
## @seealso{entropy, paddarray, stdfilt}
## @end deftypefn

function retval = entropyfilt (I, domain = true (9), padding = "symmetric", varargin)
  ## Check input
  if (nargin == 0)
    error ("entropyfilt: not enough input arguments");
  endif

  if (! isnumeric (I))
    error ("entropyfilt: I must be numeric");
  endif

  if (! isnumeric (domain) && ! islogical (domain))
    error ("entropyfilt: DOMAIN must be a logical matrix");
  endif
  domain = logical (domain);

  ## Get number of histogram bins
  if (islogical (I))
    nbins = 2;
  else
    nbins = 256;
  endif

  ## Convert to 8 or 16 bit integers if needed
  ## (accepting single, int8, int16, int32, int64, uint64 is Octave-only)
  switch (class (I))
    case {"double", "single", "int16", "int32", "int64", "uint16", "uint32", "uint64"}
      I = im2uint8 (I); # because this is what Matlab seems to do
    case {"logical", "int8", "uint8"}
      ## Do nothing
    otherwise
      error ("entropyfilt: cannot handle images of class '%s'", class (I));
  endswitch

  ## Pad image
  pad = floor (size (domain)/2);
  I = padarray (I, pad, padding, varargin {:});
  even = (round (size (domain)/2) == size (domain)/2);
  idx = cell (1, ndims (I));
  for k = 1:ndims (I)
    idx {k} = (even (k)+1):size (I, k);
  endfor
  I = I (idx {:});

  retval = __spatial_filtering__ (I, domain, "entropy", zeros (size (domain)),
                                  nbins);
endfunction

%!test
%! a = log2 (9) * ones (5, 5);
%! b = -(2*log2 (2/9) + log2 (1/9))/3;
%! a(1,2:4) = b;
%! a(5,2:4) = b;
%! a(2:4,1) = b;
%! a(2:4,5) = b;
%! c = -(4*log2 (4/9) + 4*log2 (2/9) + log2 (1/9))/9;
%! a(1,1) = c;
%! a(5,1) = c;
%! a(1,5) = c;
%! a(5,5) = c;
%! assert (entropyfilt (uint8 (magic (5)), ones (3, 3)), a, 2*eps);

%!test
%! assert (entropyfilt (uint8 (ones (10, 10))), zeros (10, 10));

## some (Matlab compatible) tests on simple 2D-images (classes double, uint8, uint16):
%!test
%! A = zeros (3,3);
%! B = ones (3,3);
%! C = [1 1 1; 2 2 2; 3 3 3];
%! D = C';
%! E = ones (3,3);
%! E(2,2) = 2;
%! F = 3 .* ones (3,3);
%! F(2,2) = 1;
%! G = [-1 2 7; -5 2 8; -7 pi 9];
%! H = [5 2 8; 1 -3 1; 5 1 0];
%! Hf = mat2gray(H);
%! X = uint8(abs(H));
%! P = [0.2 0.201 0.204; 0.202 0.203 0.205; 0.205 0.206 0.202];
%! Q = uint16([100 101 103; 100 105 102; 100 102 103]);
%! R = uint8([1 2 3 4 5; 11 12 13 14 15; 21 22 4 5 6; 5 5 3 2 1; 15 14 14 14 14]);
%! Aout = zeros (3);
%! Bout = zeros (3);
%! Cout = zeros (3);
%! Dout = zeros (3);
%! Eout = zeros (3);
%! Fout = zeros (3);
%! Gout_1 = -sum([2 7]./9.*log2([2 7]./9));
%! Gout_2 = -sum([3 6]./9.*log2([3 6]./9));
%! Gout_3 = -sum([4 5]./9.*log2([4 5]./9));
%! Gout = [Gout_1 Gout_2 Gout_3; Gout_1 Gout_2 Gout_3; Gout_1 Gout_2 Gout_3];
%! Hout_5 = -sum([2 7]./9.*log2([2 7]./9)) ;
%! Hout = [0.8916 0.8256 0.7412; 0.8256 Hout_5 0.6913; 0.7412 0.6913 0.6355];
%! Hfout_5 =  -sum([3 2 1 1 1 1]./9.*log2([3 2 1 1 1 1]./9));
%! Hfout = [2.3613 2.3296 2.2252; 2.4571 Hfout_5 2.3090; 2.4805 2.4488 2.3445];
%! Xout_5 = -sum([1 1 1 1 2 3]./9.*log2([1 1 1 1 2 3]./9));
%! Xout  = [2.3613 2.3296 2.2252; 2.4571 Xout_5 2.3090; 2.4805 2.4488 2.3445];
%! Pout_5 = -sum([1 2 6]./9.*log2([1 2 6]./9));
%! Pout = [1.1137 1.1730 1.2251; 1.1595 Pout_5 1.2774; 1.1556 1.2183 1.2635];
%! Qout = zeros(3);
%! Rout = [3.5143 3.5700 3.4871 3.4957 3.4825;
%!            3.4705 3.5330 3.4341 3.4246 3.3890;
%!            3.3694 3.4063 3.3279 3.3386 3.3030;
%!            3.3717 3.4209 3.3396 3.3482 3.3044;
%!            3.4361 3.5047 3.3999 3.4236 3.3879];
%! assert (entropyfilt (A), Aout);
%! assert (entropyfilt (B), Bout);
%! assert (entropyfilt (C), Cout);
%! assert (entropyfilt (D), Dout);
%! assert (entropyfilt (E), Eout);
%! assert (entropyfilt (F), Fout);
%! assert (entropyfilt (G), Gout, 1e-4);
%! assert (entropyfilt (H), Hout, 1e-4);
%! assert (entropyfilt (Hf), Hfout, 1e-4);
%! assert (entropyfilt (X), Xout, 1e-4);
%! assert (entropyfilt (P), Pout, 1e-4);
%! assert (entropyfilt (Q), Qout);
%! assert (entropyfilt (R), Rout, 1e-4);
