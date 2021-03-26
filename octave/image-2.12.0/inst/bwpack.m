## Copyright (C) 2018 Martin Janda <janda.martin1@gmail.com>
##
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {} bwpack (@var{bw})
## Pack binary image.
##
## Packs binary image @var{bw} into an array of 32 bit unsigned
## integers.  Each 32 elements of @var{bw} are packed into a uint32
## integer, the first row corresponding to the least significant bit
## and the 32th row corresponding to the most significant bit.
##
## Packing is performed along the first dimension (rows), so that each
## 32 rows on each column correspond to one element in the packed
## image.  @var{bw} is zero-padded if its height isn't an exact
## multiple of 32.  It is thus necessary to remember the height of the
## original image in order to retrieve it from the packed version,
## e.g. by calling @code{bwunpack}.
##
## @var{bw} is converted to logical before packing, non-zero elements
## being converted to @code{true}.
##
## @seealso{bwunpack, bitpack, bitunpack}
## @end deftypefn

function bw = bwpack (bw)
  if (nargin != 1)
    print_usage ();
  endif

  try
    bw = logical (bw);
  catch
    error ("Octave:invalid-input-arg",
           "bwpack: BW must be logical or conversible to logical")
  end_try_catch

  class_size = 32; # number of pixels packed into a single unsigned int

  dims = size (bw);
  out_nrows = ceil (dims(1) / class_size);

  in_nrows = out_nrows * class_size;
  if (in_nrows != dims(1))
    bw = resize (bw, [in_nrows dims(2:end)]);
  endif

  bw = reshape (bitpack (bw(:), "uint32"), [out_nrows dims(2:end)]);
endfunction

%!error id=Octave:invalid-fun-call bwpack ()
%!error id=Octave:invalid-input-arg bwpack ("text")

%!xtest
%! ## bug #55521
%! assert (bwpack (eye (5)), uint32 ([1 2 4 8 16]))

%!xtest
%! ## bug #55521
%! assert (bwpack (repmat (eye (4), 15, 1)),
%!         uint32 ([286331153    572662306    1145324612    2290649224
%!                  17895697     35791394      71582788     143165576]))

%!xtest
%! ## bug #55521
%! assert (bwpack (ones (3, 3, 3, 3)), repmat (uint32 (7), 1, 3, 3, 3))

%!assert (bwpack (false (0, 10)), uint32 (zeros (0, 10)))
%!assert (bwpack (false (0, 0)), uint32 (zeros (0, 0)))
%!assert (bwpack (false (32, 0)), uint32 (zeros (1, 0)))
%!assert (bwpack (false (33, 0)), uint32 (zeros (2, 0)))
%!assert (bwpack (false (0, 10, 3)), uint32 (zeros (0, 10, 3)))
%!assert (bwpack (false (33, 0, 3)), uint32 (zeros (2, 0, 3)))

## This would error in Matlab but works in Octave.  Reason is that
## `logical (i)` fails in Matlab so `bwpack (i)` makes no sense and
## fails too. However, `logical (i)` works fine in Octave so `bwpack
## (i)` must also work.
%!assert (bwpack (i), bwpack (logical (i)))
