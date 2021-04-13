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
## @deftypefn  {Function File} {} nlfilter (@var{A}, @var{block_size}, @var{func})
## @deftypefnx {Function File} {} nlfilter (@var{A}, @var{block_size}, @var{func}, @dots{})
## @deftypefnx {Function File} {} nlfilter (@var{A}, "indexed", @dots{})
## Process matrix in sliding blocks with user-supplied function.
##
## Executes the function @var{fun} on each sliding block of
## size @var{block_size}, taken from the matrix @var{A}.  Both the matrix
## @var{A}, and the block can have any number of dimensions.  This function
## is specially useful to perform sliding/moving window functions such as
## moving average.
##
## The output will have the same dimensions @var{A}, each one of its values
## corresponding to the processing of a block centered at the same
## coordinates in @var{A}, with @var{A} being padded with zeros for the
## borders (see below for indexed images).  In case any side of the block
## is of even length, the center is considered at indices
## @code{floor ([@var{block_size}/2] + 1)}.
##
## The argument @var{func} must be a function handle that takes matrices of
## size @var{block_size} as input and returns a single scalar.  Any extra
## input arguments to @code{nlfilter} are passed to @var{func} after the
## block matrix.
##
## If @var{A} is an indexed image, the second argument should be the
## string @qcode{"indexed"} so that any required padding is done correctly
## as done by @code{im2col}.
##
## @emph{Note}: if @var{func} is a column compression function, i.e., it acts
## along a column to return a single value, consider using @code{colfilt}
## which usually performs faster.  If @var{func} makes use of the colon
## operator to select all elements in the block, e.g., if @var{func} looks
## anything like @code{@@(x) sum (x(:))}, it is a good indication that
## @code{colfilt} should be used.  In addition, many sliding block operations
## have their own specific implementations (see help text of @code{colfilt}
## for a list).
##
## @seealso{blockproc, col2im, colfilt, im2col}
## @end deftypefn

function B = nlfilter (A, varargin)

  ## Input check
  [p, block_size, padval] = im2col_check ("nlfilter", nargin, A, varargin{:});

  if (nargin < p)
    print_usage ();
  endif
  func = varargin{p++};
  if (! isa (func, "function_handle"))
    error ("nlfilter: FUNC must be a function handle");
  elseif (! isscalar (func (ones (block_size, class (A)), varargin{p:nargin-1})))
    error ("nlfilter: FUNC must take a matrix of size BLOCK_SIZE and return a scalar");
  endif
  ## Remaining params are parameters to fun. We need to place them into
  ## a separate variable, we can't varargin{p:nargin-1} inside the anonymous
  ## function because nargin will have a different value there
  extra_args = varargin(p:nargin -1);

  ## Pad image as required
  padded = pad_for_sliding_filter (A, block_size, padval);

  ## Get the blocks (one per column)
  blks = im2col (padded, block_size, "sliding");

  ## New function that reshapes the array into a block before
  ## passing it to the actual user supplied function
  blk_func = @(x, z) func (reshape (blks(x:z), block_size), extra_args{:});

  ## Perform the filtering
  blk_step = 1:(rows (blks)):(rows (blks) * columns (blks));
  B = arrayfun (blk_func, blk_step, blk_step + rows (blks) -1);

  ## Back into its original shape
  B = reshape (B, size (A));

endfunction

%!demo
%! ## creates a "wide" diagonal (although it can be performed more
%! ## efficiently with "imdilate (A, true (3))")
%! nlfilter (eye (10), [3 3], @(x) any (x(:) > 0))

%!assert (nlfilter (eye (4), [2 3], @(x) sum (x(:))),
%!        [2 2 1 0
%!         1 2 2 1
%!         0 1 2 2
%!         0 0 1 1]);
%!assert (nlfilter (eye (4), "indexed", [2 3], @(x) sum (x(:))),
%!        [4 2 1 2
%!         3 2 2 3
%!         2 1 2 4
%!         4 3 4 5]);
%!assert (nlfilter (eye (4), "indexed", [2 3], @(x, y) sum (x(:)) == y, 2),
%!        logical ([0 1 0 1
%!                  0 1 1 0
%!                  1 0 1 0
%!                  0 0 0 0]));

## Check uint8 and uint16 padding (only the padding since the class of
## the output is dependent on the function and sum() always returns double)
%!assert (nlfilter (uint8 (eye (4)), "indexed", [2 3], @(x) sum (x(:))),
%!        [2 2 1 0
%!         1 2 2 1
%!         0 1 2 2
%!         0 0 1 1]);
%!assert (nlfilter (int16 (eye (4)), "indexed", [2 3], @(x) sum (x(:))),
%!        [4 2 1 2
%!         3 2 2 3
%!         2 1 2 4
%!         4 3 4 5]);

## Check if function class is preserved
%!assert (nlfilter (uint8 (eye (4)), "indexed", [2 3], @(x) int8 (sum (x(:)))),
%!        int8 ([2 2 1 0
%!               1 2 2 1
%!               0 1 2 2
%!               0 0 1 1]));

%!test
%! ## Effect of out of border elements.
%! expected = [
%!    0.5    6.0    6.0    0.5    0
%!    5.5   10.5   13.5   10.5    4.0
%!    6.5   12.5   13.5   13.5    1.5
%!   10.5   12.5   15.5   11.0    1.0
%!    5.0   10.5    6.0    1.0    0
%! ];
%! assert (nlfilter (magic (5), [3 4], @(x) median (x(:))), expected)


%!test
%! ## The center pixel of a sliding window when its length is even
%! ## sized is ceil ((size (NHOOD) +1) /2)
%! expected = [
%!   24  24  24  16  16
%!   24  24  24  22  22
%!   23  23  22  22  22
%!   25  25  25  25  22
%!   25  25  25  25  21
%! ];
%! assert (nlfilter (magic (5), [3 4], @(x) max (x(:))), expected)

## nlfilter and imdilate use a different center pixel when the
## window/nhood have an even sized length.  So to have imdilate behave
## like nlfilter, we need to flip those dimensions.
%!function dilated = imdilate_like_nlfilter (im, nhood)
%!  even_nhood_dims = find (mod (size (nhood), 2) == 0);
%!  for i = 1:even_nhood_dims
%!    im = flip (im, i);
%!  endfor
%!  dilated = imdilate (im, nhood);
%!  for i = 1:even_nhood_dims
%!    dilated = flip (dilated, i);
%!  endfor
%!endfunction

## Test N-dimensional
%!test
%! a = randi (65535, 20, 20, 20, "uint16");
%! ## extra dimensions on matrix only
%! assert (nlfilter (a, [5 5], @(x) max(x(:))), imdilate (a, ones (5)))
%! ## extra dimensions on both matrix and block
%! assert (nlfilter (a, [5 5 5], @(x) max(x(:))), imdilate (a, ones ([5 5 5])))
%! ## extra dimensions and padding
%! assert (nlfilter (a, [3 7], @(x) max(x(:))), imdilate (a, ones ([3 7])))
%! assert (nlfilter (a, [3 7 3], @(x) max(x(:))), imdilate (a, ones ([3 7 3])))

%!test
%! a = randi (65535, 15, 15, 4, 8, 3, "uint16");
%! assert (nlfilter (a, [3 4 7 5], @(x) max(x(:))),
%!         imdilate_like_nlfilter (a, ones ([3 4 7 5])))

## Just to make sure it's not a bug in imdilate
%!test
%! a = randi (65535, 15, 15, 4, 3, 8, "uint16");
%! ord = ordfiltn (a, 3, ones ([3 7 3 1 5]));
%! assert (nlfilter (a, [3 7 3 1 5], @(x) sort (x(:))(3)), ord)
%! assert (nlfilter (a, [3 7 3 1 5], @(x, y) sort (x(:))(y), 3), ord)
