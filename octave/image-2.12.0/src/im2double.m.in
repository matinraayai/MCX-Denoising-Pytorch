## Copyright (C) 2007 Søren Hauberg <soren@hauberg.org>
## Copyright (C) 2012-2014 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn  {Function File} {} im2double (@var{img})
## @deftypefnx {Function File} {} im2double (@var{img}, "indexed")
## Convert image to double precision.
##
## The conversion of @var{img} to double precision, is dependent
## on the type of input image.  The following input classes are supported:
##
## @table @samp
## @item uint8, uint16, and int16
## The whole range of values from the class (see @code{getrangefromclass})
## are scaled for the interval [0 1], e.g., if input image was uint8,
## intensity values of 0, 127, and 255, are converted to intensity of
## 0, 0.498, and 1.
##
## @item logical
## True and false values are assigned a value of 0 and 1 respectively.
##
## @item single
## Values are cast to single precision.
##
## @item double
## Returns the same image.
##
## @end table
##
## If the second argument is the string @qcode{"indexed"}, then values are
## cast to double precision, and a +1 offset is applied if input is
## an integer class.
##
## @seealso{im2bw, imcast, im2uint8, im2int16, im2single, im2uint16}
## @end deftypefn

function imout = im2double (img, varargin)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (nargin == 2 && ! strcmpi (varargin{1}, "indexed"))
    error ("im2double: second input argument must be the string \"indexed\"");
  endif
  imout = imcast (img, "double", varargin{:});
endfunction

%!assert (im2double ([1 2 3]), [1 2 3]);
%!assert (im2double (single ([1 2 3])), [1 2 3]);
%!assert (im2double (uint8 ([0 127 128 255])), [0 127/255 128/255 1]);
%!assert (im2double (uint16 ([0 127 128 65535])), [0 127/65535 128/65535 1]);
%!assert (im2double (int16 ([-32768 -32767 -32766 32767])), [0 1/65535 2/65535 1]);

%!assert (im2double (uint8 ([0 1 255]), "indexed"), [1 2 256]);
%!assert (im2double (uint16 ([0 1 2557]), "indexed"), [1 2 2558]);
%!assert (im2double ([3 25], "indexed"), [3 25]);

%!error <indexed> im2double (single ([0 1 2]), "indexed");
%!error <indexed> im2double (int16 ([17 8]), "indexed");
%!error <indexed> im2double (int16 ([-7 8]), "indexed");
%!error <indexed> im2double ([false true], "indexed");

