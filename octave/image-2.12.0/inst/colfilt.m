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
## @deftypefn  {Function File} {} colfilt (@var{A}, @var{block_size}, @var{block_type}, @var{func})
## @deftypefnx {Function File} {} colfilt (@var{A}, @var{block_size}, @var{subsize}, @var{block_type}, @var{func}, @dots{})
## @deftypefnx {Function File} {} colfilt (@var{A}, "indexed", @dots{})
## @deftypefnx {Function File} {} colfilt (@dots{}, @var{func}, @var{extra_args}, @dots{})
## Apply function to matrix blocks
##
## Executes the function @var{func} on blocks of size @var{block_size},
## taken from the matrix @var{A}.  Both the matrix @var{A}, and the block
## can have any number of dimensions.
##
## The different blocks are organized into a matrix, each block as a
## single column, and passed as the first to the function handle
## @var{func}.  Any input arguments to @code{colfilt} after @var{func}
## are passed to @var{func} after the blocks matrix.
##
## Blocks can be of two different types as defined by the string @var{block_type}:
##
## @table @asis
## @item @qcode{"distinct"}
## Each block is completely distinct from the other, with no overlapping
## elements.  @var{func} must return a matrix of exactly the same size as
## its input.
##
## @item @qcode{"sliding"}
## Each possible block of size @var{block_size} inside @var{A} is used.
## @var{func} should act along the column dimension (be a column
## compression function) and return a vector of length equal to the
## number of columns of its input.
##
## @end table
##
## The optional argument @var{subsize} divides @var{A} into smaller pieces
## before generating the matrices with one block per column in order to
## save memory.  It is currently only accepted for @sc{Matlab} compatibility.
##
## If @var{A} is an indexed image, the second argument should be the
## string @qcode{"indexed"} so that any required padding is done correctly.
## The padding value will be 0 except for indexed images of class uint8
## and uint16.
##
## This function is mostly useful to apply moving or sliding window filter
## when @var{block_type} is "sliding".  However, for many cases, specialized
## functions perform much faster.  For the following common cases, consider
## the suggested alternatives;
##
## @table @asis
## @item moving average
## A moving average filter is equivalent to convolve with a matrix
## of @code{1/@var{N}} sized @var{block_size}, where @var{N} is the total
## number of elements in a block.  Use
## @code{convn (@var{A}, (1/@var{N}) * ones (@var{block_size}) *, "same")}
##
## @item maximum or minimum
## This is the equivalent to a dilation and erosion.  Use @code{imdilate} or
## @code{imerode}.
##
## @item any or all
## Same as dilation and erosion but with logical input.  Use @code{imdilate}
## or @code{imerode} with @code{logical (@var{A})}.
##
## @item median
## Use @code{medfilt2} if @var{A} is only 2 dimensional, and @code{ordfiltn}
## with the @code{floor (prod (@var{N}/ 2)} th element, where @var{N} is the
## total number of elements in a block (add 1 if it is an even number).
##
## @item sort or nth_element
## Use @code{ordfiltn}.
##
## @item standard deviation
## Use @code{stdfilt}.
##
## @item sum
## Use a matrix of 1 to perform convolution,
## @code{convn (@var{A}, ones (@var{block_size}), "same")}
##
## @end table
##
## @seealso{bestblk, blockproc, col2im, im2col, nlfilter}
## @end deftypefn

function B = colfilt (A, varargin)

  ## Input check
  if (nargin < 4)
    print_usage ();
  endif
  [p, block_size, padval] = im2col_check ("colfilt", nargin, A, varargin{:});

  subsize = size (A);
  if (numel (varargin) < p)
    print_usage ();
  elseif (isnumeric (varargin{p}) && isvector (varargin{p}))
    subsize = varargin{p++};
    subsize = postpad (subsize, ndims (A), 1);
    subsize = min (subsize, size (A));
    subsize = max (subsize, block_size);
  endif

  ## We need at least 2 more arguments (block type and function)
  if (numel (varargin) < p +1)
    print_usage ();
  endif

  ## Next one needs to be block type
  block_type = varargin{p++};
  if (! ischar (block_type))
    error ("colfilt: BLOCK_TYPE must be a string");
  endif

  ## followed by the function
  func = varargin{p++};
  if (! isa (func, "function_handle"))
    error ("colfilt: FUNC must be a function handle");
  endif

  ## anything after this are extra arguments to func
  extra_args = varargin(p:end);

  switch (tolower (block_type))
    case "sliding"
      ## Function must return a single vector, one element per column,
      ## i.e., should act along the elements of each column.

      ## TODO for some large blocks, we may easily try to create matrix
      ##      too large for Octave (see im2col documentation about the
      ##      size). May be a good idea to split it into smaller images
      ##      even if subsize is that large, so that we never go above
      ##      sizemax ().
      ##      However, this can be tricky. After splitting the image in
      ##      smaller blocks, they can't be distinct, some parts need
      ##      to overlap otherwise when we put them back together, we'll
      ##      introduce many artifacts.

      padded = pad_for_sliding_filter (A, block_size, padval);
      cols = im2col (padded, block_size, "sliding");
      B = col2im (func (cols, extra_args{:}), block_size, size (padded), "sliding");

    case "distinct"
      ## Function must return a matrix with the same number of elements
      ## as its input, specially the same number of rows.

      ## One of the options of this function is to do the block
      ## processing already from big blocks from the original matrix
      ## in order to save memory. While this may make sense with
      ## sliding blocks, not so much here since cols will have the same
      ## size as A, and so will B.
      cols = im2col (A, block_size, "distinct");
      B = col2im (func (cols, extra_args{:}), block_size, size (A), "distinct");

    otherwise
      error ("colfilt: invalid BLOCK_TYPE `%s'.", block_type);
  endswitch

endfunction

%!demo
%! ## Perform moving average filter with a 4x4 window
%! A = magic (12)
%! colfilt (A, [4 4], "sliding", @mean)

%!test
%! A = reshape (1:36, [6 6]);
%! assert (colfilt (A, [2 2], [3 3], "sliding", @sum),
%!         conv2 (A, ones (2), "same"));

