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
## @deftypefn {Function File} {@var{S} =} stdfilt (@var{im})
## @deftypefnx{Function File} {@var{S} =} stdfilt (@var{im}, @var{domain})
## @deftypefnx{Function File} {@var{S} =} stdfilt (@var{im}, @var{domain}, @var{padding}, @dots{})
## Computes the local standard deviation in a neighbourhood around each pixel in
## an image.
##
## The standard deviation of the pixels of a neighbourhood is computed as
##
## @example
## @var{S} = sqrt ((sum (@var{x} - @var{mu}).^2)/(@var{N}-1))
## @end example
##
## where @var{mu} is the mean value of the pixels in the neighbourhood,
## @var{N} is the number of pixels in the neighbourhood. So, an unbiased estimator
## is used.
##
## The neighbourhood is defined by the @var{domain} binary mask. Elements of the
## mask with a non-zero value are considered part of the neighbourhood. By default
## a 3 by 3 matrix containing only non-zero values is used.
##
## At the border of the image, extrapolation is used. By default symmetric
## extrapolation is used, but any method supported by the @code{padarray} function
## can be used. Since extrapolation is used, one can expect a lower deviation near
## the image border.
##
## @seealso{std2, paddarray, entropyfilt}
## @end deftypefn

function retval = stdfilt (I, domain = true (3), padding = "symmetric", varargin)
  ## Check input
  if (nargin == 0)
    error ("stdfilt: not enough input arguments");
  endif

  if (! isimage (I))
    error ("stdfilt: first input must be a matrix");
  endif

  if (! isnumeric (domain) && ! islogical (domain))
    error ("stdfilt: second input argument must be a logical matrix");
  endif
  domain = logical (domain);

  ## Pad image
  pad = floor (size (domain)/2);
  I = padarray (I, pad, padding, varargin {:});
  even = (round (size (domain)/2) == size (domain)/2);
  idx = cell (1, ndims (I));
  for k = 1:ndims (I)
    idx {k} = (even (k)+1):size (I, k);
  endfor
  I = I (idx {:});

  retval = __spatial_filtering__ (I, domain, "std", zeros (size (domain)), 0);
endfunction

%!test
%! im = stdfilt (ones (5));
%! assert (im, zeros (5))

## some (Matlab compatible) tests on simple 2D-images:
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
%! A_out = [0 0 0; 0 0 0; 0 0 0];
%! B_out = [0 0 0; 0 0 0; 0 0 0];
%! C_out = repmat ([std([1 1 1 1 1 1 2 2 2])
%!                 std([1 1 1 2 2 2 3 3 3])
%!                 std([2 2 2 3 3 3 3 3 3])], [1 3]);
%! D_out = C_out';
%! E_out = (1/3) .* ones (3,3);
%! F_out = (2/3) .* ones (3,3);
%! G_out = [std([-1 -1 2 -1 -1 2 -5 -5 2]), std([-1 2 7 -1 2 7 -5 2 8]), std([2 7 7 2 7 7 2 8 8]);
%!               std([-1 -1 2 -5 -5 2 -7 -7 pi]), std([-1 2 7 -5 2 8 -7 pi 9]), std([2 7 7 2 8 8 pi 9 9]);
%!               std([-5 -5 2 -7 -7 pi -7 -7 pi]), std([-5 2 8 -7 pi 9 -7 pi 9]), std([2 8 8 pi 9 9 pi 9 9])];
%! H_out = [std([5 5 2 5 5 2 1 1 -3]), std([5 2 8 5 2 8 1 -3 1]), std([2 8 8 2 8 8 -3 1 1]);
%!                std([5 5 2 1 1 -3 5 5 1]), std([5 2 8 1 -3 1 5 1 0]), std([2 8 8 -3 1 1 1 0 0]);
%!                std([1 1 -3 5 5 1 5 5 1]), std([1 -3 1 5 1 0 5 1 0]), std([-3 1 1 1 0 0 1 0 0])];
%! assert (stdfilt (A), A_out)
%! assert (stdfilt (B), B_out)
%! assert (stdfilt (C), C_out, 4*eps)
%! assert (stdfilt (D), D_out, 4*eps)
%! assert (stdfilt (E), E_out, 4*eps)
%! assert (stdfilt (F), F_out, 4*eps)
%! assert (stdfilt (G), G_out, 4*eps)
%! assert (stdfilt (H), H_out, 4*eps)
## testing all input types
%! im = stdfilt (ones (5, 'logical'));
%! assert (im, zeros (5))
%! im = stdfilt (ones (5, 'uint8'));
%! assert (im, zeros (5))
%! assert (stdfilt (int8(H), H_out, 4*eps))
%! assert (stdfilt (uint8(H), H_out, 4*eps))
%! assert (stdfilt (int16(H), H_out, 4*eps))
%! assert (stdfilt (uint16(H), H_out, 4*eps))
%! assert (stdfilt (int32(H), H_out, 4*eps))
%! assert (stdfilt (uint32(H), H_out, 4*eps))
%! assert (stdfilt (int64(H), H_out, 4*eps))
%! assert (stdfilt (uint64(H), H_out, 4*eps))
%! assert (stdfilt (single(H), H_out, 4*eps))
