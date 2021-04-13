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
## @deftypefn  {Function File} {} ordfiltn (@var{A}, @var{nth}, @var{domain})
## @deftypefnx {Function File} {} ordfiltn (@var{A}, @var{nth}, @var{domain}, @var{S})
## @deftypefnx {Function File} {} ordfiltn (@dots{}, @var{padding})
## N dimensional ordered filtering.
##
## Ordered filter replaces an element of @var{A} with the @var{nth} element 
## element of the sorted set of neighbours defined by the logical 
## (boolean) matrix @var{domain}.
## Neighbour elements are selected to the sort if the corresponding 
## element in the @var{domain} matrix is true.
## 
## The optional variable @var{S} is a matrix of size(@var{domain}). 
## Values of @var{S} corresponding to nonzero values of domain are 
## added to values obtained from @var{A} when doing the sorting.
##
## Optional variable @var{padding} determines how the matrix @var{A} 
## is padded from the edges. See @code{padarray} for details.
## 
## @seealso{medfilt2, padarray, ordfilt2}
## @end deftypefn

## This function is based on 'ordfilt2' by Teemu Ikonen <tpikonen@pcu.helsinki.fi>
## which is released under GPLv2 or later.

function retval = ordfiltn (A, nth, domain, varargin)

  ## Check input
  if (nargin < 3)
    print_usage ();
  elseif (! isnumeric (A) && ! islogical (A))
    error ("ordfiltn: A must be a numeric or logical array");
  elseif (! isscalar (nth) || nth <= 0 || fix (nth) != nth)
    error ("ordfiltn: second input argument must be a positive integer");
  elseif (! isnumeric (domain) && ! islogical (domain))
    error ("ordfiltn: DOMAIN must be a numeric or logical array or scalar");
  elseif (isscalar (domain) && (domain <= 0 || fix (domain) != domain))
    error ("ordfiltn: third input argument must be a positive integer, when it is a scalar");
  endif

  if (isscalar (domain))
    domain = true (repmat (domain, 1, ndims (A)));
  endif
  
  if (ndims (A) != ndims (domain))
    error ("ordfiltn: first and second argument must have same dimensionality");
  elseif (any (size (A) < size (domain)))
    error ("ordfiltn: domain array cannot be larger than the data array");
  endif

  ## Parse varargin
  S = zeros (size (domain));
  padding = 0;
  for idx = 1:length(varargin)
    opt = varargin{idx};
    if (ischar (opt) || isscalar (opt))
      padding = opt;
    elseif (isnumeric (opt) && size_equal (opt, domain))
      S = opt;
    else
      error ("ordfiltn: unrecognized option from input argument #%i and class %s", 3 + idx, class (opt));
    endif
  endfor

  A = pad_for_sliding_filter (A, size (domain), padding);

  ## Perform the filtering
  retval = __spatial_filtering__ (A, logical (domain), "ordered", S, nth);

endfunction

%!shared b, f, s
%! b = [ 0  1  2  3
%!       1  8 12 12
%!       4 20 24 21
%!       7 22 25 18];
%!
%! f = [ 8 12 12 12
%!      20 24 24 24
%!      22 25 25 25
%!      22 25 25 25];
%!assert (ordfiltn (b, 9, true (3)), f);
%!
%! f = [ 1  8 12 12
%!       8 20 21 21
%!      20 24 24 24
%!      20 24 24 24];
%!assert (ordfiltn (b, 8, true (3)), f);
%!
%! f = [ 1  2  8 12
%!       4 12 20 21
%!       8 22 22 21
%!      20 24 24 24];
%!assert (ordfiltn (b, 7, true (3), "symmetric"), f);
%!
%! f = [ 1  8 12 12
%!       4 20 24 21
%!       7 22 25 21
%!       7 22 25 21];
%!assert (ordfiltn (b, 3, true (3, 1)), f);
%!
%! f = [ 1  8 12 12
%!       4 20 24 18
%!       4 20 24 18
%!       4 20 24 18];
%!assert (ordfiltn (b, 3, true (4, 1)), f);
%!
%! f = [ 4 20 24 21
%!       7 22 25 21
%!       7 22 25 21
%!       7 22 25 21];
%!assert (ordfiltn (b, 4, true (4, 1)), f);
%!
%! s = [0 0 1
%!      0 0 1
%!      0 0 1];
%! f = [ 2  8 12 12
%!       9 20 22 21
%!      21 25 24 24
%!      21 25 24 24];
%!assert (ordfiltn (b, 8, true (3), s), f);
%!
%! b(:,:,2) = b(:,:,1) - 1;
%! b(:,:,3) = b(:,:,2) - 1;
%! f(:,:,1) = [ 1  8 11 11
%!              8 20 21 21
%!              20 24 24 24
%!              20 24 24 24];
%! f(:,:,2) = [ 6 10 11 11
%!             18 22 22 22
%!             20 24 24 24
%!             20 24 24 24];
%! f(:,:,3) = [ 0  7 10 10
%!              7 19 20 20
%!             19 23 23 23
%!             19 23 23 23];
%!assert (ordfiltn (b, 25, true (3, 3, 3)), f);
