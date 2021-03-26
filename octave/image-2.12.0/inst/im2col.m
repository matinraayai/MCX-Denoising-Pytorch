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
## @deftypefn  {Function File} {} im2col (@var{A}, @var{block_size})
## @deftypefnx {Function File} {} im2col (@var{A}, @var{block_size}, @var{block_type})
## @deftypefnx {Function File} {} im2col (@var{A}, "indexed", @dots{})
## Rearrange blocks from matrix into columns.
##
## Rearranges blocks of size @var{block_size}, sampled from the matrix @var{A},
## into a serie of columns.  This effectively transforms any image into a
## 2 dimensional matrix, a block per column, which can then be passed to
## other functions that perform calculations along columns.
##
## Both blocks and matrix @var{A} can have any number of dimensions (though
## for sliding blocks, a block can't be larger than @var{A} in any dimension).
## Blocks are always accessed in column-major order (like Octave arrays are
## stored) so that a matrix can be easily reconstructed with @code{reshape}
## and @code{col2im}. For a 2 dimensional matrix, blocks are taken first from
## the top to the bottom, and then from the left to the right of the matrix.
##
## The sampling can be performed in two different ways as defined by
## @var{block_type} (defaults to @qcode{"sliding"}):
##
## @table @asis
## @item @qcode{"distinct"}
## Each block is completely distinct from the other, with no overlapping
## elements.  The matrix @var{A} is padded as required with a value of 0
## (or 1 for non-integer indexed images).
##
## @item @qcode{"sliding"}
## A single block slides across @var{A} without any padding.
##
## While this can be used to perform sliding window operations such as maximum
## and median filters, specialized functions such as @code{imdilate} and
## @code{medfilt2} will be more efficient.
##
## Note that large images being arranged in large blocks can easily exceed the
## maximum matrix size (see @code{sizemax}).  For example, a matrix @var{A} of
## size 500x500, with sliding block of size [100 100], would require a matrix
## with 2.4108e+09 elements, i.e., the number of elements in a block,
## @code{100*100}, times the number of blocks, @code{(500-10+1) * (500-10+1)}.
##
## @end table
##
## If @var{A} is an indexed image, the second argument should be the
## string @qcode{"indexed"} so that any required padding is done correctly.
## The padding value will be 0 except for indexed images of class uint8
## and uint16.
##
## @seealso{blockproc, bestblk, col2im, colfilt, nlfilter, reshape}
## @end deftypefn

## Matlab behaves weird with N-dimensional images. It ignores block_size
## elements after the first 2, and treat N-dimensional as if the extra
## dimensions were concatenated horizontally. We are performing real
## N-dimensional conversion of image blocks into colums.

function B = im2col (A, varargin)

  ## Input check
  if (nargin > 4)
    print_usage ();
  endif
  [p, block_size, padval] = im2col_check ("im2col", nargin, A, varargin{:});
  if (nargin > p)
    ## we have block_type param
    if (! ischar (varargin{p}))
      error("im2col: BLOCK_TYPE must be a string");
    endif
    block_type = varargin{p++};
  else
    block_type = "sliding";
  endif
  if (nargin > p)
    print_usage ();
  endif

  ## After all the input check, start the actual im2col. The idea is to
  ## calculate the linear indices for each of the blocks (using broadcasting
  ## for each dimension), and then reshape it with one block per column.

  switch (tolower (block_type))
    case "distinct"
      ## We may need to expand the size vector to include singletons
      size_singletons = @(x, ndim) postpad (size (x), ndim, 1);

      ## Calculate needed padding
      A_size = size_singletons (A, numel (block_size));
      sp = mod (-A_size, block_size);
      if (any (sp))
        A = padarray (A, sp, padval, "post");
      endif
      A_size = size_singletons (A, numel (block_size));

      ## Get linear indixes for the first block
      [ind, stride] = get_1st_ind (A_size, block_size);

      ## Get linear indices for all of the blocks
      blocks  = A_size ./ block_size;
      step    = stride .* block_size;
      limit   = step .* (blocks -1);
      for dim = 1:numel (A_size)
        ind = ind(:) .+ (0:step(dim):limit(dim));
      endfor
      n_blocks = prod (blocks);

    case "sliding"
      if (numel (block_size) > ndims (A))
        error ("im2col: BLOCK_SIZE can't have more elements than the dimensions of A");
      endif

      ## Get linear indixes for the first block
      [ind, stride] = get_1st_ind (size (A), block_size);

      ## Get linear indices for all of the blocks
      slides  = size (A) - block_size;
      limit   = stride .* slides;
      for dim = 1:ndims (A)
        ## We need to use bsxfun here because of
        ## https://savannah.gnu.org/bugs/?47085
        ind = bsxfun (@plus, ind(:), (0:stride(dim):limit(dim)));
      endfor
      n_blocks = prod (slides +1);

    otherwise
      error ("im2col: invalid BLOCK_TYPE `%s'.", block_type);
  endswitch

  B = reshape (A(ind(:)), prod (block_size), n_blocks);
endfunction

## Get linear indices and for the first block, and stride size per dimension
function [ind, stride] = get_1st_ind (A_size, block_size)
  stride = [1 cumprod(A_size(1:end-1))];
  limit = (block_size -1) .* stride;
  ind = 1;
  for dim = 1:numel (A_size)
    ind = ind(:) .+ (0:stride(dim):limit(dim));
  endfor
endfunction

%!demo
%! ## Divide A using distinct blocks and then reverse the operation
%! A = [ 1:10
%!      11:20
%!      21:30
%!      31:40];
%! B = im2col (A, [2 5], "distinct")
%! C = col2im (B, [2 5], [4 10], "distinct")

## test default block type
%!test
%! a = rand (10);
%! assert (im2col (a, [5 5]), im2col (a, [5 5], "sliding"))

## indexed makes no difference when sliding
%!test
%! a = rand (10);
%! assert (im2col (a, [5 5]), im2col (a, "indexed", [5 5]))

%!error <BLOCK_TYPE> im2col (rand (20), [2 5], 10)
%!error <BLOCK_TYPE> im2col (rand (20), [2 5], "wrong_block_type")
%!error im2col (rand (10), [5 5], "sliding", 5)
%!error im2col (rand (10), "indexed", [5 5], "sliding", 5)

%!shared B, A, Bs, As, Ap, Bp0, Bp1, Bp0_3s
%! v   = [1:10]';
%! r   = reshape (v, 2, 5);
%! B   = [v v+20  v+40 v+10  v+30 v+50];
%! A   = [r r+10; r+20 r+30; r+40 r+50];
%! As  = [ 1  2  3  4  5
%!         6  7  8  9 10
%!        11 12 13 14 15];
%! b1  = As(1:2, 1:4)(:);
%! b2  = As(2:3, 1:4)(:);
%! b3  = As(1:2, 2:5)(:);
%! b4  = As(2:3, 2:5)(:);
%! Bs  = [b1, b2, b3, b4];
%! Ap  = A(:, 1:9);
%! Bp1 = Bp0 = B;
%! Bp0(9:10, 4:6) = 0;
%! Bp1(9:10, 4:6) = 1;
%! Bp0_3s = Bp0;
%! Bp0_3s(11:30, :) = 0;

## test distinct block type
%!assert (im2col (A, [2 5], "distinct"), B);

## padding for distinct
%!assert (im2col (Ap, [2 5], "distinct"), Bp0);
%!assert (im2col (Ap, [2 5 3], "distinct"), Bp0_3s);
%!assert (im2col (Ap, "indexed", [2 5], "distinct"), Bp1);
%!assert (im2col (uint8  (Ap), "indexed", [2 5], "distinct"), uint8  (Bp0));
%!assert (im2col (uint16 (Ap), "indexed", [2 5], "distinct"), uint16 (Bp0));
%!assert (im2col (int16  (Ap), "indexed", [2 5], "distinct"), int16  (Bp1));
%!assert (im2col (uint32 (Ap), "indexed", [2 5], "distinct"), uint32 (Bp1));

## Always return correct class
%!assert (im2col (uint8   (A),  [2 5], "distinct"), uint8   (B));
%!assert (im2col (single  (A),  [2 5], "distinct"), single  (B));
%!assert (im2col (logical (A),  [2 5], "distinct"), logical (B));
%!assert (im2col (uint8   (As), [2 4], "sliding"),  uint8   (Bs));
%!assert (im2col (single  (As), [2 4], "sliding"),  single  (Bs));
%!assert (im2col (logical (As), [2 4], "sliding"),  logical (Bs));

## test sliding block type
%!assert (im2col (As, [2 4], "sliding"), Bs);
%!assert (im2col (As, [3 5], "sliding"), As(:));

## Test N-dimensional
%!test
%! A = randi (9, 10, 9, 5);
%!assert (convn (A, ones (3, 3, 3), "valid"),
%!        reshape (sum (im2col (A, [3 3 3])), [8 7 3]));
%!
%! A = randi (9, 10, 9, 5, 7);
%!assert (convn (A, ones (3, 3, 3), "valid"),
%!        reshape (sum (im2col (A, [3 3 3])), [8 7 3 7]));
%!assert (convn (A, ones (3, 4, 3), "valid"),
%!        reshape (sum (im2col (A, [3 4 3])), [8 6 3 7]));
%!assert (convn (A, ones (3, 5, 3, 2), "valid"),
%!        reshape (sum (im2col (A, [3 5 3 2])), [8 5 3 6]));

## Corner case for Matlab compatibility -- bug #46774
%!assert (im2col (1:8, [2 1]), zeros (2, 0))
