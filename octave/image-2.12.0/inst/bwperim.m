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
## @deftypefn  {Function File} {} bwperim (@var{bw})
## @deftypefnx {Function File} {} bwperim (@var{bw}, @var{conn})
## Find perimeter of objects in binary images.
##
## Values from the matrix @var{bw} are considered part of an object perimeter
## if their value is non-zero and is connected to at least one zero-valued
## element, or to the outside of @var{bw}.
##
## Element connectivity @var{conn}, to define the size of objects, can be
## specified with a numeric scalar (number of elements in the neighborhood):
##
## @table @samp
## @item 4 or 8
## for 2 dimensional matrices;
## @item 6, 18 or 26
## for 3 dimensional matrices;
## @end table
##
## or with a binary matrix representing a connectivity array.  Defaults to
## @code{conndef (ndims (@var{bw}), "minimal")} which is equivalent to
## @var{conn} of 4 and 6 for 2 and 3 dimensional matrices respectively.
##
## @seealso{bwarea, bwboundaries, imerode, mmgrad}
## @end deftypefn

function varargout = bwperim (bw, conn)

  if (nargin < 1 || nargin > 2)
    print_usage ();
  endif

  if (! isnumeric (bw) && ! islogical (bw))
    error("bwperim: BW must be a numeric or logical matrix");
  endif
  bw = logical (bw);

  if (nargin < 2)
    conn = conndef (ndims (bw), "minimal");
  else
    conn = conndef (conn);
  endif

  ## Recover the elements that would get removed by erosion
  perim = (! imerode (bw, conn)) & bw;

  ## Get the borders back which are removed during erosion
  ## FIXME this is a bit too convoluted and not elegant at all. I am also
  ##        unsure if it is correct for N dimensional stuff and unusual
  ##        connectivities.  We should probably be using the output from
  ##        bwboundaries() but bwboundaries() seems buggy in the case of
  ##        holes.
  tmp_idx       = repmat ({":"}, [1 ndims(perim)]);
  tmp_conn_idx  = repmat ({":"}, [1 ndims(conn)]);
  p_size  = size (perim);
  for dim = 1:min (ndims (perim), ndims (conn))
    conn_idx = tmp_conn_idx;
    conn_idx{dim} = [1 3];
    if (! any (conn(conn_idx{:})(:)))
      continue
    endif

    idx = tmp_idx;
    idx{dim} = [1 p_size(dim)];
    perim(idx{:}) = bw(idx{:});
  endfor

  if (nargout > 0)
    varargout{1} = perim;
  else
    imshow (perim);
  endif
endfunction

%!test
%! in = [ 1   1   1   1   0   1   1   0   1   1
%!        1   1   0   1   1   1   1   1   1   0
%!        1   1   1   0   1   1   1   1   1   1
%!        1   1   1   1   0   1   1   1   0   1
%!        1   1   1   0   1   1   1   1   1   0
%!        1   1   1   1   1   1   0   1   0   1
%!        1   1   1   1   1   1   1   1   1   0
%!        1   1   1   1   1   1   1   1   1   1
%!        1   1   1   1   1   1   0   0   1   1
%!        1   1   1   1   0   1   0   1   1   0];
%!
%! out = [1   1   1   1   0   1   1   0   1   1
%!        1   1   0   1   1   0   0   1   1   0
%!        1   0   1   0   1   0   0   0   1   1
%!        1   0   0   1   0   1   0   1   0   1
%!        1   0   1   0   1   0   1   0   1   0
%!        1   0   0   1   0   1   0   1   0   1
%!        1   0   0   0   0   0   1   0   1   0
%!        1   0   0   0   0   0   1   1   0   1
%!        1   0   0   0   1   1   0   0   1   1
%!        1   1   1   1   0   1   0   1   1   0];
%! assert (bwperim (in), logical (out))
%! assert (bwperim (in, 4), logical (out))
%!
%! out = [1   1   1   1   0   1   1   0   1   1
%!        1   1   0   1   1   1   1   1   1   0
%!        1   1   1   0   1   1   0   1   1   1
%!        1   0   1   1   0   1   0   1   0   1
%!        1   0   1   0   1   1   1   1   1   0
%!        1   0   1   1   1   1   0   1   0   1
%!        1   0   0   0   0   1   1   1   1   0
%!        1   0   0   0   0   1   1   1   1   1
%!        1   0   0   1   1   1   0   0   1   1
%!        1   1   1   1   0   1   0   1   1   0];
%! assert (bwperim (in, 8), logical (out))
%!
%! out = [1   1   1   1   0   1   1   0   1   1
%!        1   0   0   0   0   1   0   0   1   0
%!        1   0   0   0   0   0   0   1   0   1
%!        1   0   1   0   0   0   0   0   0   1
%!        1   0   0   0   0   1   0   1   0   0
%!        1   0   0   0   1   0   0   0   0   1
%!        1   0   0   0   0   0   0   1   0   0
%!        1   0   0   0   0   1   1   0   0   1
%!        1   0   0   1   0   1   0   0   1   1
%!        1   1   1   1   0   1   0   1   1   0];
%! assert (bwperim (in, [1 0 0; 0 1 0; 0 0 1]), logical (out))

## test that any non-zero value is valid (even i and Inf)
%!test
%! in = [ 0   0   0   0   0   0   0
%!        0   0   5   0   0   1   9
%!        0 Inf   9   7   0   0   0
%!        0 1.5   5   7   1   0   0
%!        0 0.5  -1  89   i   0   0
%!        0   4  10  15   1   0   0
%!        0   0   0   0   0   0   0];
%! out = [0   0   0   0   0   0   0
%!        0   0   1   0   0   1   1
%!        0   1   0   1   0   0   0
%!        0   1   0   0   1   0   0
%!        0   1   0   0   1   0   0
%!        0   1   1   1   1   0   0
%!        0   0   0   0   0   0   0];
%! assert (bwperim (in), logical (out))

## test for 3D
%!test
%! in = reshape (magic(16), [8 8 4]) > 50;
%! out(:,:,1) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   1   1   1   0   1
%!    0   1   1   1   1   1   0   1
%!    1   1   0   1   1   1   1   1
%!    1   1   1   1   1   1   1   1
%!    1   1   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! out(:,:,2) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   0   1   1   0   1
%!    0   1   0   0   0   1   0   1
%!    1   0   1   0   0   0   1   1
%!    1   0   0   1   0   1   0   1
%!    1   0   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! out(:,:,3) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   0   1   1   0   1
%!    0   1   0   0   0   1   0   1
%!    1   0   0   0   0   0   1   1
%!    1   0   0   1   0   1   0   1
%!    1   0   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! out(:,:,4) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   1   1   1   0   1
%!    0   1   1   1   1   1   0   1
%!    1   1   1   1   1   1   1   1
%!    1   1   1   1   1   1   1   0
%!    1   1   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! assert (bwperim (in), logical (out))
%!
%! out(:,:,1) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   1   1   1   0   1
%!    0   1   1   1   1   1   0   1
%!    1   1   0   1   1   1   1   1
%!    1   1   1   1   1   1   1   1
%!    1   1   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! out(:,:,2) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   1   1   1   0   1
%!    0   1   1   0   0   1   0   1
%!    1   1   1   1   0   1   1   1
%!    1   0   1   1   1   1   1   1
%!    1   0   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! out(:,:,3) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   1   1   1   0   1
%!    0   1   0   0   0   1   0   1
%!    1   1   0   0   0   1   1   1
%!    1   0   1   1   1   1   1   1
%!    1   0   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! out(:,:,4) = [
%!    1   1   0   1   0   1   1   1
%!    0   1   1   1   1   1   0   1
%!    0   1   1   1   1   1   0   1
%!    1   1   1   1   1   1   1   1
%!    1   1   1   1   1   1   1   0
%!    1   1   1   0   1   0   1   1
%!    1   1   1   0   1   0   1   1
%!    1   0   1   1   1   1   1   0];
%! assert (bwperim (in, 18), logical (out))

%!error bwperim ("text")
%!error bwperim (rand (10), 5)
%!error bwperim (rand (10), "text")

%!test
%! a = false (5);
%! a(1:4,2:4) = true;
%!
%! p = false (5);
%! p(1:4,[2 4]) = true;
%! assert (bwperim (a, [0 0 0; 1 1 1; 0 0 0]), p)

## This is not a bug.  Since connectivity defaults to maximum for the
## number of dimensions, a single slice will be displayed completely
## (this is obvious but very easy to forget)
%!test
%! a = false (8, 8, 5);
%! a(4:5,4:5,2:4) = true;
%! a(2:7,2:7,3) = true;
%! assert (bwperim (a, 26), a)
%!
%! ## It is easy to forget that is correct
%! b = a;
%! b(4:5, 4:5, 3) = false;
%! assert (bwperim (a), b)
%!
%! c = a;
%! c(3:6,3:6,3) = false;
%! assert (bwperim (a, 4), c)

## test dimensions of length 1 (1x1, Nx1, etc) (bug #50153)
%!test
%! conn_self = logical ([0 0 0; 0 1 0; 0 0 0]);
%! assert (bwperim (true), true)
%! assert (bwperim (true, conn_self), false)
%! assert (bwperim (true (1, 6)), true (1, 6))
%! assert (bwperim (true (1, 6), conn_self), false (1, 6))
%! assert (bwperim (true (6, 1)), true (6, 1))
%!
%! bw_3d = true (1, 1, 6);
%! assert (bwperim (bw_3d), bw_3d)
%! assert (bwperim (bw_3d, conn_self), false (1, 1, 6))
%! assert (bwperim (bw_3d, true (3)), bw_3d)
%!
%! perim_3d = bw_3d;
%! perim_3d(1, 1, 2:end-1) = false;
%! conn_3d = false (3, 3, 3);
%! conn_3d(2, 2, :) = true;
%! assert (bwperim (true (1, 1, 6), conn_3d), perim_3d)
