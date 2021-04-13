## Copyright (C) 2018 Martin Janda <janda.martin1@gmail.com>
## Copyright (C) 2018 David Miguel Susano Pinto <carandraug@octave.org>
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
## @deftypefn  {} {} bwunpack (@var{bwp})
## @deftypefnx {} {} bwunpack (@var{bwp}, @var{m})
## Unpack binary image.
##
## Each row of the packed binary image @var{bwp}, represented as an
## uint32 matrix, is unpacked into 32 rows of logical values.  The
## unpacking is done such that the least significant bit of the first
## element in @var{bwp} maps to the first element in the unpacked
## image, and the most significant bit to the 32th element.
##
## The unpacked image will be a logical array with @var{m} rows.
## The length of the other dimensions will be the same as @var{bwp}.
## If @var{m} is not specified, it will unpack all bits in @var{bwp},
## otherwise the extra bits will be considered padding resulting
## from the packing.  See the help text for @code{bwpack} for
## details.
##
## @seealso{bwpack, bitpack, bitunpack}
## @end deftypefn

function bw = bwunpack (bwp, m)
  if (nargin < 1)
    print_usage ();
  endif

  if (! isa (bwp, "uint32"))
    error ("Octave:invalid-input-arg", "bwunpack: BWP must be an uint32 array");
  endif

  class_size = 32; # number of elements packed into a uint32

  if (nargin < 2)
    m = rows (bwp) * class_size;
  elseif (m < 0 || fix (m) != m)
    error ("Octave:invalid-input-arg",
           "bwunpack: M must be a non-negative integer");
  elseif (m > (rows (bwp) * class_size))
    error ("Octave:invalid-input-arg",
           ["bwunpack: M must not be larger than the number of bits " ...
              "on each column"]);
  endif

  packed_rows = rows (bwp) * class_size;
  dims = size (bwp);
  bw = reshape (bitunpack (bwp), [packed_rows dims(2:end)]);
  bw(m+1:end,:) = []; # remove padding if any
endfunction

%!error id=Octave:invalid-fun-call bwunpack ()
%!error <BWP must be an uint32 array> bwunpack (uint8 (1))
%!error <M must be a non-negative integer> bwunpack (uint32 (1), -1)
%!error <M must be a non-negative integer> bwunpack (uint32 (1), 4.2)

%!xtest
%! ## bug #55521
%! assert (bwunpack (uint32 (2.^[0:31])), logical (eye (32)))

%!xtest
%! ## bug #55521
%! assert (bwunpack (uint32 (repmat (7, [1 3 3 3])), 3), true (3, 3, 3, 3))

%!assert (bwunpack (uint32 (zeros (0, 0))), false (0, 0))
%!assert (bwunpack (uint32 (zeros (0, 0)), 0), false (0, 0))
%!assert (bwunpack (uint32 (zeros (0, 5)), 0), false (0, 5))
%!assert (bwunpack (uint32 (zeros (0, 5, 7)), 0), false (0, 5, 7))
%!assert (bwunpack (uint32 (zeros (1, 0))), false (32, 0))
%!assert (bwunpack (uint32 (zeros (2, 0, 7))), false (64, 0, 7))
%!assert (bwunpack (uint32 (zeros (2, 0, 7))), false (64, 0, 7))
%!assert (bwunpack (uint32 (zeros (2, 0, 7)), 60), false (60, 0, 7))

## Unpacking more bytes than what's available on the input needs to be
## an error.  This works in Matlab but it's their bug and the result
## is not even reproducible.
%!error <M must not be larger than the number of bits on each column>
%!      bwunpack (uint32 (1), 1042)
