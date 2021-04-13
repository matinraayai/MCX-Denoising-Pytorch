## Copyright (C) 2015 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see
## <http:##www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {[@var{quant}, @var{idx}] =} imquantize (@var{img}, @var{levels})
## @deftypefnx {Function File} {[@dots{}] =} imquantize (@var{img}, @var{levels}, @var{values})
## Quantize image with multiple threshold levels and values.
##
## This function reduces the number of unique values in @var{img} by
## performing multiple thresholds, one for each element in @var{levels},
## and assigning it values from @var{values}.
##
## The output @var{quant} is the quantized image, of same size as @var{img}
## and same class as @var{values}.  @var{idx} are the indices into
## @var{values} such that @code{@var{quant} == @var{values}(@var{idx})}.
##
## For the simpler case where @var{values} is not defined, it defaults
## to @code{1:N+1} and this function becomes equivalent to:
##
## @example
## out = ones (size (@var{img}));
## for i = 1:numel(@var{levels})
##   out(@var{img} > @var{levels}(i)) += 1;
## endfor
## @end example
##
## So that for example:
##
## @example
## >> img = [1 2 3; 4 5 6; 7 8 9];
## >> imquantize (img, [3 6])
##    @result{} ans =
##
##          1   1   1
##          2   2   2
##          3   3   3
## @end example
##
## For the more general case, when @var{values} is defined, this function
## is equivalent to:
##
## @example
## @var{levels} = [-Inf @var{levels} Inf];
## out = zeros (size (@var{img}), class (@var{values}));
## for i = 1:numel(@var{values})
##   out(@var{img} > @var{levels}(i) & @var{img} <= @var{levels}(i+1)) = @var{values}(i);
## endfor
## @end example
##
## So that for example:
##
## @example
## >> img = [1 2 3; 4 5 6; 7 8 9];
## >> imquantize (img, [3 6], [0 5 8])
##    @result{} ans =
##
##          0   0   0
##          5   5   5
##          8   8   8
## @end example
##
## @seealso{gray2ind, ind2gray, ind2rgb, intlut, label2rgb, lookup, rgb2ind}
## @end deftypefn

function [quant, idx] = imquantize (img, levels, values)

  if (nargin < 2 || nargin > 3)
    print_usage ();
  elseif (! isimage (img))
    error ("imquantize: IMG must be an image - numeric array with real values");
  endif

  validateattributes (levels, {"numeric"}, {"nondecreasing", "vector"},
                      "imquantize", "LEVELS");

  ## Add eps to levels because lookup does
  ##    "table(idx(i)) <= y(i) < table(idx(i+1))"
  ## But we want:
  ##    "table(idx(i)) < y(i) <= table(idx(i+1))"
  levels = double (levels);
  if (isfloat (img))
    ## Beware when img is of class single.  The comparison in lookup()
    ## will be done in the class of the image if it is float.
    levels += eps (cast (levels, class (img)));
  else
    levels += eps (levels);
  endif

  idx = lookup ([-Inf; levels(:); Inf], img);

  if (nargin < 3)
    quant = idx;
  else
    if (! isvector (values) || ! isnumeric (values))
      error ("imquantize: VALUES must be a numeric vector")
    elseif (numel (values) != numel (levels) +1)
      error ("imquantize: VALUES must have 1 element more than LEVELS");
    endif
    quant = values(idx);
  endif
endfunction

%!error <LEVELS must be nondecreasing>
%!  imquantize (rand (5), [3 4 2 5])
%!error <VALUES must be a numeric vector>
%!  imquantize (rand (5), [1 2 3], "foo")
%!error <VALUES must have 1 element more than LEVELS>
%!  imquantize (rand (5), [1 2 3 4], 1:6)
%!error <VALUES must have 1 element more than LEVELS>
%!  imquantize (rand (5), [1 2 3 4], 1:2)

%!test
%! img = [-inf 0 10000000; -100000 -3 1/1000000; 5 5 10];
%! [q, q_idx] = imquantize (img, 5);
%! assert (q, [1 1 2; 1 1 1; 1 1 2])
%! assert (q_idx, q)

%!test
%! img = [1:10; 11:20; 21:30; 31:40; 41:50; 51:60; 61:70];
%!
%! expected_q = [
%!     0    0    0    0    0    1    1    1    1    1
%!     1    1    1    1    1    5    5    5    5    5
%!     5    5    5    5    5   10   10   10   10   10
%!    20   20   20   20   20   20   20   20   20   20
%!    30   30   30   30   30   30   30   30   30   30
%!    30   30   30   30   30   30   30   30   30   30
%!    15   15   15   15   15   15   15   15   15   15];
%!
%! expected_q_idx = [
%!    1   1   1   1   1   2   2   2   2   2
%!    2   2   2   2   2   3   3   3   3   3
%!    3   3   3   3   3   4   4   4   4   4
%!    5   5   5   5   5   5   5   5   5   5
%!    6   6   6   6   6   6   6   6   6   6
%!    6   6   6   6   6   6   6   6   6   6
%!    7   7   7   7   7   7   7   7   7   7];
%!
%! [q, q_idx] = imquantize (img, [5 15 25 30 40 60], [0 1 5 10 20 30 15]);
%! assert (q, expected_q)
%! assert (q_idx, expected_q_idx)
%!
%! [q, q_idx] = imquantize (single (img), [5 15 25 30 40 60],
%!                          [0 1 5 10 20 30 15]);
%! assert (q, expected_q)
%! assert (q_idx, expected_q_idx)
%!
%! [q, q_idx] = imquantize (uint8 (img), [5 15 25 30 40 60],
%!                          [0 1 5 10 20 30 15]);
%! assert (q, expected_q)
%! assert (q_idx, expected_q_idx)
%!
%! [q, q_idx] = imquantize (uint8 (img), uint8 ([5 15 25 30 40 60]),
%!                          [0 1 5 10 20 30 15]);
%! assert (q, expected_q)
%! assert (q_idx, expected_q_idx)
%!
%! [q, q_idx] = imquantize (uint8 (img), uint8 ([5 15 25 30 40 60]),
%!                          uint8 ([0 1 5 10 20 30 15]));
%! assert (q, uint8 (expected_q))
%! assert (q_idx, expected_q_idx)

## Test output class
%!test
%! img = randi ([0 255], 10, "uint8");
%! [q, q_idx] = imquantize (img, [50 100 150 200]);
%! assert (class (q), "double")
%! assert (class (q_idx), "double")
%!
%! [q, q_idx] = imquantize (img, [50 100 150 200], uint16 ([5 7 8 9 2]));
%! assert (class (q), "uint16")
%! assert (class (q_idx), "double")
%!
%! [q, q_idx] = imquantize (img, [50 100 150 200], uint8 ([5 7 8 9 2]));
%! assert (class (q), "uint8")
%! assert (class (q_idx), "double")

## A test with non-ordered pixel values, tested against ordered
%!test
%! img = [1:10; 11:20; 21:30; 31:40; 41:50; 51:60; 61:70].';
%! r_idx = reshape (randperm (numel (img)), size (img));
%!
%! [quant, quant_idx] = imquantize (img, [5 15 25 30 40 60]);
%! [quant_r, quant_r_idx] = imquantize (img(r_idx), [5 15 25 30 40 60]);
%!
%! assert (imquantize (img(r_idx), [5 15 25 30 40 60]), quant(r_idx))
%! assert (quant_r, quant_r_idx)
