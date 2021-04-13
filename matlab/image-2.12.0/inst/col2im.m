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
## @deftypefn  {Function File} {} col2im (@var{B}, @var{block_size}, @var{A_size})
## @deftypefnx {Function File} {} col2im (@var{B}, @var{block_size}, @var{A_size}, @var{block_type})
## Rearrange block columns back into matrix.
##
## Rearranges columns of the matrix @var{B}, representing blocks of size
## @var{block_size} from a matrix of size @var{A_size}, back into its
## original size (usually close to @var{A_size}.  This function is most
## useful as reverse operation to @code{im2col}.
##
##
## Blocks are assumed to be from one of two types as defined by
## @var{block_type} (defaults to @qcode{"sliding"}):
##
## @table @asis
## @item @qcode{"distinct"}
## Each column of @var{B} is assumed to be distinct blocks, with no
## overlapping elements, of size @var{block_size}, to rebuild a matrix of
## size @var{A_size}.  Any padding that may have been required to form
## @var{B} from a matrix of @var{A_size}, is removed accordingly.
##
## @item @qcode{"sliding"}
## This reshapes @var{B} into a matrix of size
## @code{@var{A_size} - @var{block_size} +1}.  Sliding blocks are most useful
## to apply a sliding window filter with functions that act along columns.
## In this situation, @var{B} is usually a row vector, so that if
## @var{block_size} is [1 1], @var{A_SIZE} will be the size of the output
## matrix.  When converting a matrix into blocks with @code{im2col}, there
## will be less blocks to account to borders, so if @var{block_size} is the
## same in both @code{col2im} and @code{im2col}, @var{A_size} can be the size
## out the output from @code{im2col}.
##
## @end table
##
## Blocks are assumed to have been from a matrix, the same direction elements
## are organized in an Octave matrix (top to bottom, then left to right), and
## the direction that blocks are taken in @code{im2col}.
##
## @group
## @example
## ## Get distinct blocks of size [2 3] from A into columns, and
## ## put them back together into the original position
## A = reshape (1:24, [4 6])
## B = im2col (A, [2 3], "distinct")
## col2im (B, [2 3], [4 6], "distinct")
## @end example
##
## @example
## ## Get sliding blocks of size [2 3] from A into columns, calculate
## ## the mean of each block (mean of each column), and reconstruct A.
## ## This is the equivalent to a sliding window filter and ignoring
## ## borders.
## A = reshape (1:24, [4 6])
## B = im2col (A, [2 3], "sliding")
## C = mean (B);
## col2im (C, [1 1], [3 4], "sliding")
## @end example
## @end group
##
## @seealso{blockproc, bestblk, colfilt, im2col, nlfilter, reshape}
## @end deftypefn

function A = col2im (B, block_size, A_size, block_type = "sliding")

  if (nargin < 3 || nargin > 4)
    print_usage ();
  elseif (! isnumeric (B) && ! islogical (B))
    error ("col2im: B must be a numeric of logical matrix or vector");
  elseif (! isnumeric (block_size) || ! isvector (block_size))
    error("col2im: BLOCK_SIZE must be a numeric vector");
  elseif (! isnumeric (A_size) || ! isvector (A_size))
    error("col2im: A_SIZE must be a numeric vector");
  elseif (! ischar (block_type))
    error ("col2im: BLOCK_TYPE must be a string");
  endif

  ## Make sure dimensions are row vectors
  block_size  = block_size(:).';
  A_size      = A_size(:).';

  ## expand size to include singleton dimensions if required
  block_size(end+1:numel (A_size)) = 1;
  A_size(end+1:numel (block_size)) = 1;

  switch (tolower (block_type))
    case "distinct"
      ## A_size is out_size for distinct blocks
      if (prod (block_size) != rows (B))
        error (["col2im: number of rows in B, must equal number of " ...
                "elements per block (prod (BLOCK_SIZE))"]);
      elseif (numel (B) < prod (A_size))
        error ("col2im: not enough elements in B for a matrix of A_SIZE");
      endif

      ## Calculate the number of blocks accross each dimension, and
      ## how much of padding we will need to remove (we calculate the
      ## number of cumulative elements by dimension, so the first
      ## element is the number of rows that are pad, the second
      ## second is the number of columns that are pad times the number
      ## of elements in a column, and so on for other dimensions)
      blocks  = ceil (A_size ./ block_size);
      padding = mod (-A_size, block_size) .* [1 cumprod(A_size(1:end-1))];

      cum_blk_size = [1 cumprod(block_size(1:end-1))];
      cum_blocks   = [1 cumprod(blocks(1:end-1))];

      ## End of each block dimension in the column
      end_blk = ceil (cum_blk_size .* (block_size -1));

      ## How much in the columns we need to shift to move to the
      ## next block per dimension.
      stride = rows (B) * cum_blocks;

      ## Last block for each dimension
      last_blk = stride .* (blocks -1);

      ind = 1;
      for dim = 1:numel(A_size)
        ind = ind(:) .+ (0:cum_blk_size(dim):end_blk(dim));
        ind = ind(:) .+ (0:stride(dim):last_blk(dim));
        ind(end+1-padding(dim):end) = []; # remove padding
      endfor
      A = reshape (B(ind(:)), A_size);

    case "sliding"
      out_size = A_size - block_size +1;
      if (prod (out_size) != numel (B))
        error ("col2im: can't resize B in matrix sized (A_SIZE - BLOCK_SIZE +1)");
      endif
      A = reshape (B, out_size);

    otherwise
      error ("col2im: invalid BLOCK_TYPE `%s'.", block_type);
  endswitch

endfunction

%!demo
%! ## Divide A using distinct blocks and then reverse the operation
%! A = [ 1:10
%!      11:20
%!      21:30
%!      31:40];
%! B = im2col (A, [2 5], "distinct")
%! C = col2im (B, [2 5], [4 10], "distinct")

%!demo
%! ## Get sliding blocks of size from A into columns, calculate the
%! ## mean of each block (mean of each column), and reconstruct A
%! ## after a median filter.
%! A = reshape (1:24, [4 6])
%! B = im2col (A, [2 3], "sliding")
%! C = mean (B);
%! col2im (C, [1 1], [3 4], "sliding")

%!error <BLOCK_TYPE> col2im (ones (10), [5 5], [10 10], "wrong_block_type");
%!error <resize B>   col2im (ones (10), [1 1], [ 7  7], "sliding");
%!error <rows in B>  col2im (ones (10), [3 3], [10 10], "distinct")
%!error <rows in B>  col2im (ones (10), [5 5], [10 11], "distinct");

## test sliding
%!assert (col2im (sum (im2col (magic (10), [3 3], "sliding")), [1 1], [8 8]),
%!        convn (magic (10), ones (3, 3), "valid"));

%!test
%! B = ones (1, (10-2+1)*(7-3+1));
%! A = ones ((10-2+1), (7-3+1));
%! assert (col2im (B, [2 3], [10 7]), A);
%!
%! ## same but different classes
%! assert (col2im (int16   (B), [2 3], [10 7]), int16   (A));
%! assert (col2im (single  (B), [2 3], [10 7]), single  (A));
%! assert (col2im (logical (B), [2 3], [10 7]), logical (A));

## test default to sliding
%!test
%! a = rand (10)(:);
%! assert (col2im (a, [1 1], [10 10]), col2im (a, [1 1], [10 10], "sliding"))

## test distinct
%!shared A, B
%! v  = [1:10]';
%! r  = reshape (1:10, [2 5]);
%! B  = [v  v+10  v+20  v+30  v+40  v+50];
%! A  = [r    r+30
%!       r+10 r+40
%!       r+20 r+50];
%! assert (col2im (B, [2 5], [6 10], "distinct"), A);

## respect different classes
%!assert (col2im (int16   (B), [2 5], [6 10], "distinct"), int16   (A));
%!assert (col2im (logical (B), [2 5], [6 10], "distinct"), logical (A));
%!assert (col2im (single  (B), [2 5], [6 10], "distinct"), single  (A));

## Test for columns with padding
%!test
%! a = rand (10, 8);
%! b = im2col (a, [5 5], "distinct");
%! assert (col2im (b, [5 5], [10 8], "distinct"), a);
%!
%! a = rand (8);
%! b = im2col (a, [5 5], "distinct");
%! assert (col2im (b, [5 5], [8 8], "distinct"), a);

## Test N-dimensional
%!shared a, b
%! ## Same number of multiple dimensions
%! a = rand (10, 10, 10);
%! b = im2col (a, [5 5 5], "distinct");
%!assert (col2im (b, [5 5 5], [10 10 10], "distinct"), a);
%!
%! ## Different number of dimensions
%! a = rand (10, 10, 10);
%! b = im2col (a, [5 5], "distinct");
%!assert (col2im (b, [5 5], [10 10 10], "distinct"), a);
%!
%! ## Removing padding from multiple dimensions
%! a = rand (10, 10, 7);
%! b = im2col (a, [5 5 3], "distinct");
%!assert (col2im (b, [5 5 3], [10 10 7], "distinct"), a);
%!
%! a = rand (10, 10, 7);
%! b = im2col (a, [5 5 5 2], "distinct");
%!assert (col2im (b, [5 5 5 2], [10 10 7], "distinct"), a);
