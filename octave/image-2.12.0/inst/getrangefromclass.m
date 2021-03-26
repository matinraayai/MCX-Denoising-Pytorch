## Copyright (C) 2011-2013 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn {Function File} {@var{range} =} getrangefromclass (@var{img})
## Return display range of image.
##
## For a given image @var{img}, returns the 1x2 element matrix @var{range}
## with the display range (minimum and maximum display values) for an
## image of that class.
##
## Images of different classes have different display ranges, the ranges
## of values that Octave will interpret between black to white.  For an
## integer image, the range is from @code{intmin} to @code{intmax} of
## that class; for images of class logical, single, or double, the range
## is [0 1].
##
## Note that @var{range} will be of class double, independently of the class
## of @var{img}.
##
## @example
## @group
## getrangefromclass (ones (5)) # note that class is 'double'
##      @result{} [0   1]
## getrangefromclass (logical (ones (5)))
##      @result{} [0   1]
## getrangefromclass (int8 (ones (5)))
##      @result{} [-128  127]
## @end group
## @end example
##
## @seealso{intmin, intmax, bitmax}
## @end deftypefn

function r = getrangefromclass (img)

  if (nargin != 1)
    print_usage ();
  elseif (! isimage (img))
    error ("getrangefromclass: IMG must be an image");
  endif

  cl = class (img);
  if (isinteger (img))
    r = [intmin(cl) intmax(cl)];
  elseif (any (strcmp (cl, {"single", "double", "logical"})))
    r = [0 1];
  else
    error ("getrangefromclass: unrecognized image class `%s'", cl)
  endif
  r = double (r);
endfunction

%!shared img
%! img = ones (5);
%!assert (getrangefromclass (double (img)), [0 1]);      # double returns [0 1]
%!assert (getrangefromclass (single (img)), [0 1]);      # single returns [0 1]
%!assert (getrangefromclass (logical (img)), [0 1]);     # logical returns [0 1]
%!assert (getrangefromclass (int8 (img)), [-128 127]);   # checks int
%!assert (getrangefromclass (uint8 (img)), [0 255]);     # checks unit
%!fail ("getrangefromclass ('string')");                 # fails with strings
%!fail ("getrangefromclass ({3, 4})");                   # fails with cells
