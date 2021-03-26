## Copyright (C) 2000 Teemu Ikonen <tpikonen@pcu.helsinki.fi>
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
## @deftypefn  {Function File} {} ordfilt2 (@var{A}, @var{nth}, @var{domain})
## @deftypefnx {Function File} {} ordfilt2 (@var{A}, @var{nth}, @var{domain}, @var{S})
## @deftypefnx {Function File} {} ordfilt2 (@dots{}, @var{padding})
## Two dimensional ordered filtering.
##
## This function exists only for @sc{matlab} compatibility as is just a wrapper
## to the @code{ordfiltn} which performs the same function on N dimensions.  See
## @code{ordfiltn} help text for usage explanation.
##
## @seealso{medfilt2, padarray, ordfiltn}
## @end deftypefn

function A = ordfilt2 (A, nth, domain, varargin)

  if (nargin < 3)
    print_usage ();
  elseif (ndims (A) > 2 || ndims (domain) > 2 )
    error ("ordfilt2: A and DOMAIN are limited to 2 dimensinos. Use `ordfiltn' for more")
  endif
  A = ordfiltn (A, nth, domain, varargin{:});

endfunction

%!test
%! order = 3;
%! domain = ones (3);
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
%! B_out = [0 0 0; 0 1 0; 0 0 0];
%! C_out = [0 0 0; 0 1 0; 0 0 0];
%! D_out = [0 0 0; 0 1 0; 0 0 0];
%! E_out = [0 0 0; 0 1 0; 0 0 0];
%! F_out = [0 0 0; 0 3 0; 0 0 0];
%! G_out = [0 0 0; -1 -1 0; 0 0 0];
%! H_out = [0 0 0; 0 1 0; 0 0 0];
%! assert (ordfilt2 (A, order, domain), A_out);
%! assert (ordfilt2 (B, order, domain), B_out);
%! assert (ordfilt2 (C, order, domain), C_out);
%! assert (ordfilt2 (D, order, domain), D_out);
%! assert (ordfilt2 (E, order, domain), E_out);
%! assert (ordfilt2 (F, order, domain), F_out);
%! assert (ordfilt2 (G, order, domain), G_out);
%! assert (ordfilt2 (H, order, domain), H_out);
