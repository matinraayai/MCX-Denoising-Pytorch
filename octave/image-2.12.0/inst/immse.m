## Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
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
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} immse (@var{x}, @var{y})
## Compute mean squared error.
##
## Calculates the mean squared error (MSE), between the arrays @var{x} and
## @var{y}.  @var{x} and @var{y} must be of same size and class.
##
## The returned value will be a scalar double, unless @var{x} and @var{y}
## are of class single in which case, it returns a scalar single.
##
## @seealso{psnr}
## @end deftypefn

function [mse] = immse (x, y)

  if (nargin != 2)
    print_usage ();
  elseif (! size_equal (x, y))
    error ("immse: X and Y must be of same size");
  elseif (! strcmp (class (x), class (y)))
    error ("immse: X and Y must have same class");
  end

  if (isinteger (x))
    x = double (x);
    y = double (y);
  endif

  err = x - y;
  mse = sumsq (err(:)) / numel (err);

endfunction

%!error <same size> immse (rand (10), rand (12))
%!error <same class> immse (uint8 ([0 1 2 3]), uint16 ([0 1 2 3]))
%!error <same class> immse (double ([0 1 2 3]), single ([0 1 2 3]))

%!assert (immse (magic (5), magic (5)), 0)
%!assert (immse (single (magic (5)), single (magic (5))), single (0))
%!assert (immse (uint8 (magic (5)), uint8 (magic (5))), 0)

