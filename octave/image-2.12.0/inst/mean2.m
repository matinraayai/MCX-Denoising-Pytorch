## Copyright (C) 2000 Kai Habel <kai.habel@gmx.de>
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
## @deftypefn {Function File} {} mean2 (@var{I})
## Compute mean value of array.
##
## While the function name suggests that it computes the mean value of
## a 2D array, it will actually computes the mean value of an entire
## array.  It is equivalent to @code{mean (I(:))}.
##
## The return value will be of class double independently of the input
## class.
##
## @seealso{mean, std2}
## @end deftypefn

function m = mean2 (I)
  if (nargin != 1)
    print_usage();
  endif
  m = mean (I(:));
endfunction

## Corner cases for Matlab compatibility (bug #51144)

%!test
%! ## This throws a division by zero warning which Matlab does not, but
%! ## that's because Matlab does not throw such warnings in the first
%! ## place.  Octave does, so we do not turn the warning off.
%! warning ("off", "Octave:divide-by-zero", "local");
%! assert (mean2 ([]), NaN)

%!assert (mean2 (logical ([1 1; 0 0])), 0.5)
%!assert (mean2 (ones (3, 3, 3)), 1)
%!assert (mean2 (i), i)
%!assert (mean2 ([1 i]), [0.5+0.5i])
%!assert (mean2 (speye (3)), sparse (1/3))
