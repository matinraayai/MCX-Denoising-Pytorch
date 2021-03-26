## Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
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
## @deftypefn  {Function File} {@var{blk_size} =} bestblk (@var{IMS})
## @deftypefnx {Function File} {@var{blk_size} =} bestblk (@var{IMS}, @var{max})
## @deftypefnx {Function File} {[@var{Mb}, @var{Nb}, @dots{}] =} bestblk (@dots{})
## Calculate block best size for block processing.
##
## Given a matrix of size @var{IMS}, calculates the largest size for distinct
## blocks @var{blk_size}, that minimize padding and is smaller than or equal to
## @var{k} (defaults to 100) 
##
## The output @var{blk_size} is a row vector for the block size.  If there are
## multiple output arguments, the number of rows is assigned to the first
## (@var{Mb}), and the number of columns to the second (@var{Nb}), etc.
##
## To determine @var{blk_size}, the following is performed for each
## dimension:
##
## @enumerate
## @item
## If dimension @var{IMS} is less or equal than @var{k}, it returns the
## dimension value.
##
## @item
## If not, find the highest value between @code{min (dimension/10, k/2)}
## which minimizes padding.
##
## @end enumerate
##
## @seealso{blockproc, col2im, im2col}
## @end deftypefn

function [varargout] = bestblk (ims, k = 100)

  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (! isnumeric (ims) || ! isvector (ims) || any (ims(:) < 1))
    error("bestblk: IMS must be a numeric vector of positive integers.");
  elseif (numel (ims) < 2)
    error ("bestblk: IMS must have at least 2 elements");
  elseif (! isnumeric (k) || ! isscalar (k) || k < 1)
    error ("bestblk: K must be a positive scalar");
  endif

  ims = floor (ims(:).');
  k   = floor (k);

  out = zeros (size (ims));
  for dim = 1:numel (ims)
    if (ims(dim) <= k)
      out(dim) = ims(dim);
    else
      possible = k:-1:min (ims(dim) /10, k /2);
      [~, ind] = min (mod (-ims(dim), possible));
      out(dim) = possible(ind);
    endif
  endfor

  if (nargout <= 1)
    varargout{1} = out;
  else
    varargout = mat2cell (out', ones (1, numel (out)));
  endif

endfunction

%!demo
%! siz = bestblk ([200; 10], 50);
%! disp (siz)

%!error <numeric vector> bestblk ("string")
%!error <positive scalar> bestblk ([100 200], "string")
%!error <2 elements> bestblk ([100], 5)

%!assert (bestblk ([ 10  12],   2), [  2   2]);
%!assert (bestblk ([ 10  12],   3), [  2   3]);
%!assert (bestblk ([300 100], 150), [150 100]);
%!assert (bestblk ([256 128],  17), [ 16  16]);

## make sure we really pick the highest one
%!assert (bestblk ([ 17  17],   3), [  3   3]);

## Test default
%!assert (bestblk ([230 470]), bestblk ([230 470], 100))

## Test N-dimensional
%!assert (bestblk ([10 12 10], 3), [2 3 2]);
%!assert (bestblk ([ 9 12  9], 3), [3 3 3]);
%!assert (bestblk ([10 12 10 11], 5), [5 4 5 4]);
