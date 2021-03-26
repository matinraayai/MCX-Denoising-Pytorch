## Copyright (C) 2012-2016 Carnë Draug <carandraug@octave.org>
## Copyright (C) 2012 Pantxo Diribarne <pantxo@dibona>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} checkerboard ()
## @deftypefnx {Function File} {} checkerboard (@var{side})
## @deftypefnx {Function File} {} checkerboard (@var{side}, @var{size})
## @deftypefnx {Function File} {} checkerboard (@var{side}, @var{M}, @var{N})
## @deftypefnx {Function File} {} checkerboard (@var{side}, @var{M}, @var{N}, @var{P}, @dots{})
## Create checkerboard.
##
## The checkerboard is created by repeating a tile pattern and creating
## a block matrix of size @var{size}, or @var{M}x@var{N}.  The tile pattern
## itself is made of four squares of @var{side} pixels wide.  Note how the
## number of squares is twice of @var{size}.
##
## The tile pattern is white on black on the right side, and grey on black
## on the left side of the matrix.
##
## At the simplest case, a 2 by 2 pattern with squares of 1 pixel side.
##
## @example
## checkerboard (1, [2 2])
## @result{}
##    0.0   1.0   0.0   0.7
##    1.0   0.0   0.7   0.0
##    0.0   1.0   0.0   0.7
##    1.0   0.0   0.7   0.0
## @end example
##
## Defaults to 4x4 tiles 10 pixels wide.
##
## N-Dimensional checkerboards are supported with @var{size} of any length
## or specifying dimension length arguments, i.e., @var{M}, @var{N}, @var{P},
## @dots{}.
##
## @seealso{ndgrid, repmat}
## @end deftypefn

function [board] = checkerboard (side = 10, varargin)

  if (nargin > 0 && ! (isscalar (side) && isnumeric (side)
                       && side == fix (side) && side >= 0))
    error ("checkerboard: SIDE must be a non-negative integer")
  endif

  if (numel (varargin) == 0)
    nd = 2;
    lengths = [4 4];
  else
    if (any (! cellfun ("isnumeric", varargin)))
      error ("checkerboard: SIZE or MxNx... list must be numeric");
    endif
    first_var = varargin{1};
    if (numel (varargin) == 1)
      if (isscalar (first_var)) # checkerboard (SIDE, M)
        lengths = [first_var first_var];
      else # checkerboard (SIDE, [M N P ...])
        lengths = first_var;
      endif
    else # checkerboard (SIDE, M, N, P, ...)
      if (any (cellfun ("numel", varargin) > 1))
        error ("checkerboard: M, N, P, ... must be numeric scalars")
      endif
      lengths = cell2mat (varargin);
    endif
  endif

  if (! isvector (lengths) || any (lengths < 0)
      || any (lengths != fix (lengths)))
    error ("checkerboard: SIZE or MxNx... list must be non-negative integer")
  endif
  nd = numel (lengths);

  grids = nthargout (1:nd, @ndgrid, linspace (-1, 1, 2*side));
  tile = grids{1};
  for d = 2:nd
    tile .*= grids{d};
  endfor
  tile = tile < 0;
  board = double (repmat (tile, lengths));

  ## Set left side of checkerboard to grey on black (0.7 instead of 1).
  ##
  ## ideally we would have an option to specify the dimension to do this
  ## (instead of dimension two - left and right).  However, then we can't
  ## easily differentiate between length of last dimension and the number
  ## of dimension to do this, so leave it up to the user to permute the
  ## dimensions after.
  left_idx = repmat ({":"}, 1, nd);
  nc = columns (board);
  left_idx{2} = (nc/2 +1):nc;
  board(left_idx{:}) *= 0.7;

endfunction

%!demo
%! ## Simplest case, default checkerboard size:
%! ##     8 by 8 checkerboard, with squares 10 pixel wide
%! board = checkerboard ();
%! imshow (board)

%!demo
%! ## Simplest case, default checkerboard size:
%! ##     8 by 16 checkerboard, with squares 5 pixel wide
%! board = checkerboard (5, 4, 8);
%! imshow (board)


## Special case of "SIZE == 0" still respects number of dimensions.
%!assert (checkerboard (0), zeros (0, 0))
%!assert (checkerboard (0, 3), zeros (0, 0))
%!assert (checkerboard (0, 2, 4), zeros (0, 0))
%!assert (checkerboard (0, 2, 4, 3), zeros (0, 0, 0))
%!assert (checkerboard (0, 2, 4, 3, 2), zeros (0, 0, 0, 0))

## Other special cases that leads to empty checkerboards.
%!assert (checkerboard (1, 4, 2, 3, 0), zeros (8, 4, 6, 0))
%!assert (checkerboard (1, 4, 0, 3, 2), zeros (8, 0, 6, 4))
%!assert (checkerboard (2, 4, 0, 3, 2), zeros (16, 0, 12, 8))

%!test
%! out = zeros (80);
%! i1 = ((1:20:80) .+ (0:9)')(:);
%! i2 = ((11:20:80) .+ (0:9)')(:);
%! out(i1, i2) = 1;
%! out(i2, i1) = 1;
%! i1r = ((41:20:80) .+ (0:9)')(:);
%! i2r = ((51:20:80) .+ (0:9)')(:);
%! out(i2, i1r) = 0.7;
%! out(i1, i2r) = 0.7;
%! assert (checkerboard (), out)
%! assert (checkerboard (10, 4, 4), out)
%! assert (checkerboard (10, [4 4]), out)
%! assert (checkerboard (10, [4; 4]), out)

%!test
%! out = zeros (8);
%! out(2:2:8, 1:2:8) = 1;
%! out(1:2:8, 2:2:8) = 1;
%! out(1:2:8, 6:2:8) = 0.7;
%! out(2:2:8, 5:2:8) = 0.7;
%! assert (checkerboard (1), out)
%! assert (checkerboard (1, 4), out)
%! assert (checkerboard (1, 4, 4), out)
%! assert (checkerboard (1, [4 4]), out)

%!test
%! out = zeros (10);
%! out(2:2:10, 1:2:10) = 1;
%! out(1:2:10, 2:2:10) = 1;
%! out(1:2:10, 6:2:10) = 0.7;
%! out(2:2:10, 7:2:10) = 0.7;
%! assert (checkerboard (1, 5), out)
%! assert (checkerboard (1, 5, 5), out)
%! assert (checkerboard (1, [5 5]), out)

%!test
%! out = zeros (20);
%! out([1:4:20 2:4:20], [3:4:20 4:4:20]) = 1;
%! out([3:4:20 4:4:20], [1:4:20 2:4:20]) = 1;
%! out([1:4:20 2:4:20], [11:4:20 12:4:20]) = 0.7;
%! out([3:4:20 4:4:20], [13:4:20 14:4:20]) = 0.7;
%! assert (checkerboard (2, 5), out)
%! assert (checkerboard (2, 5, 5), out)
%! assert (checkerboard (2, [5 5]), out)

%!test
%! out = zeros (4, 4, 4);
%! out([1 3], 1, [1 3]) = 1;
%! out([2 4], 2, [1 3]) = 1;
%! out([1 3], 2, [2 4]) = 1;
%! out([2 4], 1, [2 4]) = 1;
%! out([1 3], 3, [1 3]) = 0.7;
%! out([2 4], 4, [1 3]) = 0.7;
%! out([1 3], 4, [2 4]) = 0.7;
%! out([2 4], 3, [2 4]) = 0.7;
%! assert (checkerboard (1, [2 2 2]), out)
%! assert (checkerboard (1, 2, 2, 2), out)

%!test
%! out = zeros (8, 8, 8);
%! out([1 2 5 6], [1 2], [1 2 5 6]) = 1;
%! out([3 4 7 8], [3 4], [1 2 5 6]) = 1;
%! out([1 2 5 6], [3 4], [3 4 7 8]) = 1;
%! out([3 4 7 8], [1 2], [3 4 7 8]) = 1;
%! out([1 2 5 6], [5 6], [1 2 5 6]) = 0.7;
%! out([3 4 7 8], [7 8], [1 2 5 6]) = 0.7;
%! out([1 2 5 6], [7 8], [3 4 7 8]) = 0.7;
%! out([3 4 7 8], [5 6], [3 4 7 8]) = 0.7;
%! assert (checkerboard (2, [2 2 2]), out)
%! assert (checkerboard (2, 2, 2, 2), out)
