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
## @deftypefn  {Function File} {} medfilt2 (@var{A})
## @deftypefnx {Function File} {} medfilt2 (@var{A}, @var{nhood})
## @deftypefnx {Function File} {} medfilt2 (@var{A}, [@var{M} @var{N}])
## @deftypefnx {Function File} {} medfilt2 (@dots{}, @var{pad})
## Two dimensional median filtering.
##
## Replaces elements of @var{A} with the median of their neighbours as defined
## by the true elements of logical matrix @var{nhood} or by a matrix of size
## @var{M} by @var{N}.  The default @var{nhood} is a 3 by 3 matrix of true
## elements.
##
## @example
## @group
## ## median filtering specifying neighborhood dimensions
## medfilt2 (img)         # default is [3 3]
## medfilt2 (img, [3 1])  # a 3x1 vector
## medfilt2 (img, [5 5])  # 5 element wide square
## @end group
##
## @group
## ## median filtering specifying neighborhood
## medfilt2 (img, true (5)) # same as [5 5]
## nhood = logical ([0 1 0
##                   1 1 1
##                   0 1 0]);
## medfilt2 (img, nhood)    # 3 element wide cross
## @end group
## @end example
##
## The optional variable @var{pad} defines the padding used in augmenting
## the borders of @var{A}.  See @code{padarray} for details.
## @seealso{ordfilt2, ordfiltn}
## @end deftypefn

function retval = medfilt2 (A, varargin)

  if (nargin < 1 || nargin > 3)
    print_usage ();
  elseif (! isimage (A))
    error ("medfilt2: A must be a real matrix")
  endif

  ## defaults
  padding = "zeros";
  domain  = true (3, 3);

  for idx = 1:numel (varargin)
    opt = varargin{idx};
    if (ischar (opt) || isscalar (opt))
      padding = opt;
    elseif (isnumeric (opt) || islogical (opt))
      ## once we have removed the ability of selecting domain with a non logical
      ## matrix, we can use it to make this N dimensional domain = [3 3 3].
      if (isvector (opt) && ! islogical (opt) && numel (opt) == 2)
        domain = true (opt);
      else
        if (! islogical (opt))
          persistent warned = false;
          if (! warned)
            warned = true;
            warning ("medfilt2: to specify filter instead of dimensions use a matrix of logical class. This usage will be removed from future releases of the image package.");
          endif
        endif
        domain = logical (opt);
      endif
    else
      error ("medfilt2: unrecognized option of class %s", class (opt))
    endif
  endfor

  ## TODO this should probably be implemented as an option of
  ##      __spatial_filtering__ where median() cold be called directly
  n = nnz (domain);
  if ((n - 2*floor(n/2)) == 0) % n even - more work
    nth = floor (n/2);
    a = ordfiltn (A, nth, domain, padding);
    b = ordfiltn (A, nth + 1, domain, padding);
    retval = a./2 + b./2; # split into two divisions to avoid overflow on integer data
  else
    nth = floor (n/2) + 1;
    retval = ordfiltn (A, nth, domain, padding);
  endif

endfunction

%!shared b, f
%! b = [ 0  1  2  3
%!       1  8 12 12
%!       4 20 24 21
%!       7 22 25 18];
%! f = [ 0  1  2  0
%!       1  4 12  3
%!       4 12 20 12
%!       0  7 20  0];
%!assert (medfilt2 (b), f);
%!
%! f = [ 0  1  2  3
%!       1  8 12 12
%!       4 20 24 18
%!       4 20 24 18];
%!assert (medfilt2 (b, true (3, 1)), f);
%!assert (medfilt2 (b, [3 1]), f);
%!
%! f = [ 1  8 10 10
%!       1  8 12 12
%!       4 20 24 18
%!       7 20 24 18];
%!assert (medfilt2 (b, [3 1], 10), f);
%!assert (medfilt2 (b, 10, [3 1]), f);
%!
%! f = [ 0.5  4.5  7.0  7.5
%!       2.5 14.0 18.0 15.0
%!       2.5 14.0 18.0 15.0
%!       2.0 10.0 12.0  9.0];
%!assert (medfilt2 (b, true (4, 1)), f);
%!assert (medfilt2 (b, [4 1]), f);

%!test
%! A = zeros (3, 3);
%! B = ones (3, 3);
%! C = [1 1 1; 2 2 2; 3 3 3];
%! D = C';
%! E = ones (3, 3);
%! E(2,2) = 2;
%! F = 3 .* ones (3, 3);
%! F(2,2) = 1;
%! G = [-1 2 7; -5 2 8; -7 pi 9];
%! H = [5 2 8; 1 -3 1; 5 1 0];
%! A_out = [0 0 0; 0 0 0; 0 0 0];
%! B_out = [0 1 0; 1 1 1; 0 1 0];
%! C_out =  [0 1 0; 1 2 1; 0 2 0];
%! D_out = [0 1 0; 1 2 2; 0 1 0];
%! E_out = [0 1 0; 1 1 1; 0 1 0];
%! F_out = [0 3 0; 3 3 3; 0 3 0];
%! G_out = [0 0 0; 0 2 2; 0 0 0];
%! H_out = [0 1 0; 1 1 0; 0 0 0];
%! assert (medfilt2 (A), A_out);
%! assert (medfilt2 (B), B_out);
%! assert (medfilt2 (C), C_out);
%! assert (medfilt2 (D), D_out);
%! assert (medfilt2 (E), E_out);
%! assert (medfilt2 (F), F_out);
%! assert (medfilt2 (G), G_out);
%! assert (medfilt2 (H), H_out);
