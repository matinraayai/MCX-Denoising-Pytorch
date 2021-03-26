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
## @deftypefn {Function File} {@var{R} =} rangefilt (@var{im})
## @deftypefnx{Function File} {@var{R} =} rangefilt (@var{im}, @var{domain})
## @deftypefnx{Function File} {@var{R} =} rangefilt (@var{im}, @var{domain}, @var{padding}, @dots{})
## Computes the local intensity range in a neighbourhood around each pixel in
## an image.
##
## The intensity range of the pixels of a neighbourhood is computed as
##
## @example
## @var{R} = max (@var{x}) - min (@var{x})
## @end example
##
## where @var{x} is the value of the pixels in the neighbourhood,
##
## The neighbourhood is defined by the @var{domain} binary mask. Elements of the
## mask with a non-zero value are considered part of the neighbourhood. By default
## a 3 by 3 matrix containing only non-zero values is used.
##
## At the border of the image, extrapolation is used. By default symmetric
## extrapolation is used, but any method supported by the @code{padarray} function
## can be used.
##
## @seealso{paddarray, entropyfilt, stdfilt}
## @end deftypefn

function retval = rangefilt (I, domain = true (3), padding = "symmetric", varargin)
  ## Check input
  if (nargin == 0)
    error ("rangefilt: not enough input arguments");
  endif

  if (! isnumeric (I) && ! islogical (I))
    error ("rangefilt: I must be a numeric or logical array");
  endif

  if (! isnumeric (domain) && ! islogical (domain))
    error ("rangefilt: DOMAIN must be a logical array");
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

  retval = __spatial_filtering__ (I, domain, "range", zeros (size (domain)), 0);
endfunction

%!test
%! im = rangefilt (ones (5));
%! assert (im, zeros (5));

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
%! C_out = [1 1 1; 2 2 2; 1 1 1];
%! D_out = [1 2 1; 1 2 1; 1 2 1];
%! E_out = [1 1 1; 1 1 1; 1 1 1];
%! F_out = [2 2 2; 2 2 2; 2 2 2];
%! G_out = [7 13 6; 7+pi 16 7; 7+pi 16 7];
%! H_out = [8 11 11; 8 11 11; 8 8 4];
%! assert (rangefilt (A), A_out)
%! assert (rangefilt (B), B_out)
%! assert (rangefilt (C), C_out)
%! assert (rangefilt (D), D_out)
%! assert (rangefilt (E), E_out)
%! assert (rangefilt (F), F_out)
%! assert (rangefilt (G), G_out, eps)
%! assert (rangefilt (H), H_out)
